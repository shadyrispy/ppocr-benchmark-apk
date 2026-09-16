package com.example.ppocrbench;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends Activity {

    private static final String TAG = "PpocrBench";
    private static final String[] ENGINES = {"lw", "ort", "mnn", "ncnn"};
    private static final String[] MODELS = {"det", "cls", "rec"};

    // Every vendor knob each engine actually exposes. lw has none at all, so it
    // only ever appears once -- that is the finding, not an omission.
    private static final String[][] CASES = {
            {"lw",   "cpu"},
            {"ort",  "cpu"},
            {"ort",  "nnapi"},
            {"ort",  "nnapi-fp16"},
            {"mnn",  "cpu"},
            {"mnn",  "opencl"},
            {"mnn",  "opencl-fp16"},
            {"mnn",  "vulkan"},
            {"mnn",  "vulkan-fp16"},
            {"ncnn", "cpu"},
            {"ncnn", "vulkan"},
            {"ncnn", "vulkan-fp16"},
            // int8 quantization (only the engines whose offline quantizer
            // produced a working model in this pass: ORT dynamic int8 and
            // ncnn table-based int8). MNN int8 calibration config was not
            // validated, so it is omitted -- see results/REPORT.md.
            {"ort",  "int8"},
            {"ncnn", "int8"},
    };

    private TextView logView;
    private volatile boolean running = false;
    private int threads = 1;
    private int iters = 20;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        logView = findViewById(R.id.log);

        spawnExtract();

        setupSpinner(R.id.threads, new String[]{"1", "2", "4"}, v -> threads = Integer.parseInt(v));
        setupSpinner(R.id.iters, new String[]{"10", "20", "30"}, v -> iters = Integer.parseInt(v));

        findViewById(R.id.runForward).setOnClickListener(v -> startBench(false));
        findViewById(R.id.runPipeline).setOnClickListener(v -> startBench(true));

        log("model: PP-OCRv6 tiny   abi: "
                + (BuildConfig_ABI.SUPPORTED.length > 0 ? BuildConfig_ABI.SUPPORTED[0] : "?"));
        log("visible cores: " + Runtime.getRuntime().availableProcessors());
        log("tap 纯 forward 或 det 流水线 to start.");

        // Headless trigger for scripted runs:
        //   adb shell am start -n com.example.ppocrbench/.MainActivity \
        //       -a com.example.ppocrbench.RUN --ei threads 1 --ei iters 20 [--ez pipeline true]
        Intent intent = getIntent();
        autoStart = intent.getBooleanExtra("autostart", false);
        autoPipeline = intent.getBooleanExtra("pipeline", false);
        autoThreads = intent.getIntExtra("threads", 0);
        autoIters = intent.getIntExtra("iters", 0);
        autoConfigs = intent.getBooleanExtra("configs", false);
        // Optional restriction: only bench this one engine. Lets each engine
        // run in its OWN process so a native crash (e.g. an EP that segfaults
        // under memory pressure) cannot take down the other engines' data.
        singleEngine = intent.getStringExtra("single");
    }

    private boolean autoStart = false;
    private boolean autoPipeline = false;
    private int autoThreads = 0;
    private int autoIters = 0;
    private boolean autoConfigs = false;
    private String singleEngine = null;

    private static final class BuildConfig_ABI {
        static final String[] SUPPORTED = android.os.Build.SUPPORTED_ABIS;
    }

    private void setupSpinner(int id, String[] items, Picker picker) {
        Spinner s = findViewById(id);
        s.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, items));
        s.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, View v, int pos, long id) {
                picker.pick(items[pos]);
            }
            @Override public void onNothingSelected(AdapterView<?> p) { }
        });
    }

    private interface Picker { void pick(String v); }

    private File outDir() {
        File d = new File(getExternalFilesDir(null), "bench-results");
        if (!d.exists() && !d.mkdirs()) {
            Log.w(TAG, "cannot create " + d);
        }
        return d;
    }

    private void spawnExtract() {
        new Thread(() -> {
            File base = new File(getFilesDir(), "bench");
            copyAssetDir("bench", base, base);
            BenchNative.setDataDir(base.getAbsolutePath());
            Log.i(TAG, "assets extracted to " + base.getAbsolutePath());
            if (autoStart) {
                if (autoThreads > 0) threads = autoThreads;
                if (autoIters > 0) iters = autoIters;
                runOnUiThread(() -> startBench(autoPipeline));
            }
        }, "asset-extract").start();
    }

    private void copyAssetDir(String rel, File file, File base) {
        try {
            String[] names = getAssets().list(rel);
            if (names == null || names.length == 0) {
                try (InputStream in = getAssets().open(rel);
                     FileOutputStream out = new FileOutputStream(file)) {
                    byte[] buf = new byte[1 << 16];
                    int n;
                    while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
                }
                return;
            }
            if (!file.exists() && !file.mkdirs()) {
                Log.w(TAG, "mkdir failed " + file);
            }
            for (String n : names) {
                copyAssetDir(rel + "/" + n, new File(file, n), base);
            }
        } catch (Exception e) {
            Log.e(TAG, "copyAssetDir " + rel, e);
        }
    }

    private void startBench(boolean pipeline) {
        if (running) return;
        running = true;
        log(pipeline ? "\n=== det pipeline (threads=" + threads + ", iters=" + iters + ") ==="
                : "\n=== pure forward (threads=" + threads + ", iters=" + iters + ") ===");
        new Thread(() -> {
            StringBuilder sb = new StringBuilder();
            try {
                if (pipeline) {
                    for (String[] c : sweep()) {
                        String json = BenchNative.runDetPipeline(c[0], c[1], threads, 3, iters);
                        Log.i(TAG, "pipeline " + c[0] + "/" + c[1] + " -> " + json);
                        sb.append(c[0]).append('/').append(c[1]).append("/det  ")
                          .append(json).append('\n');
                        log(json);
                    }
                } else {
                    // Each engine now runs in its own process (single= intent),
                    // so per-engine CPU-boost / thermal drift is already balanced
                    // across the four runs. Keep a fixed forward order for logs.
                    String[][] cases = sweep();
                    for (String[] c : cases) {
                        for (String m : MODELS) {
                            String json = BenchNative.runForwardCase(c[0], m, c[1],
                                    threads, 3, iters);
                            Log.i(TAG, "forward " + c[0] + "/" + c[1] + "/" + m + " -> " + json);
                            sb.append(c[0]).append('/').append(c[1]).append('/').append(m)
                              .append("  ").append(json).append('\n');
                            log(json);
                        }
                    }
                }
            } catch (Throwable t) {
                Log.e(TAG, "bench failed", t);
                log("ERROR " + t);
            }
            String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
            File out = new File(outDir(), "bench-" + stamp + ".json");
            try (FileOutputStream os = new FileOutputStream(out)) {
                os.write(sb.toString().getBytes("UTF-8"));
            } catch (Exception e) {
                Log.e(TAG, "write results", e);
            }
            log("saved -> " + out.getAbsolutePath());
            running = false;
        }, "bench-run").start();
    }

    // "configs" sweeps every vendor knob; otherwise keep the original
    // one-case-per-engine fp32 sweep so old numbers stay comparable.
    private String[][] sweep() {
        String[][] all = autoConfigs ? CASES : baseCases();
        if (singleEngine != null && !singleEngine.isEmpty()) {
            ArrayList<String[]> out = new ArrayList<>();
            for (String[] c : all) {
                if (c[0].equals(singleEngine)) out.add(c);
            }
            return out.toArray(new String[0][]);
        }
        return all;
    }

    private String[][] baseCases() {
        String[][] out = new String[ENGINES.length][];
        for (int i = 0; i < ENGINES.length; i++) {
            out[i] = new String[]{ENGINES[i], "cpu"};
        }
        return out;
    }

    private void log(String msg) {
        Log.i(TAG, msg);
        runOnUiThread(() -> logView.append(msg + "\n"));
    }
}
