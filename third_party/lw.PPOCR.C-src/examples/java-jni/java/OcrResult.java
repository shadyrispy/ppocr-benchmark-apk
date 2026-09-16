import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable detailed OCR result using source-image coordinates. */
public final class OcrResult {
    private final int width;
    private final int height;
    private final List<OcrLine> lines;

    public OcrResult(int width, int height, List<OcrLine> lines) {
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("invalid image dimensions");
        if (lines == null) throw new NullPointerException("lines");
        this.width = width;
        this.height = height;
        this.lines = Collections.unmodifiableList(new ArrayList<OcrLine>(lines));
    }
    public int getWidth() { return width; }
    public int getHeight() { return height; }
    public List<OcrLine> getLines() { return lines; }
}
