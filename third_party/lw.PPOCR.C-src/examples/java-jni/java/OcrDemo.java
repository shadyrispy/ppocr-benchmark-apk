/** Console entry point for the Java/JVM OCR example. */
public final class OcrDemo {
    private OcrDemo() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 4) {
            System.err.println("Usage: OcrDemo <models-directory> <image-file> [workers] [cls]");
            System.exit(2);
        }

        String modelDirectory = args[0];
        String imagePath = args[1];
        int workers = args.length >= 3 ? Integer.parseInt(args[2]) : 0;
        boolean useCls = args.length >= 4 && Boolean.parseBoolean(args[3]);

        long started = System.nanoTime();
        OcrResult detailed;
        try (NativeOcr ocr = new NativeOcr(modelDirectory, useCls, workers)) {
            detailed = ocr.recognizeFileDetailed(imagePath);
            String[] legacy = ocr.recognizeFile(imagePath);
            if (legacy.length != detailed.getLines().size())
                throw new IllegalStateException("legacy and detailed line counts differ");
            for (int index = 0; index < legacy.length; ++index) {
                if (!legacy[index].equals(detailed.getLines().get(index).getText()))
                    throw new IllegalStateException("legacy and detailed text differs at line " + index);
            }
            for (OcrLine line : detailed.getLines()) {
                float[] box = line.getBox();
                if (box.length != 8) {
                    throw new IllegalStateException("OCR box must contain 8 coordinates");
                }
                for (int coordinate = 0; coordinate < box.length; ++coordinate) {
                    float value = box[coordinate];
                    boolean xCoordinate = (coordinate & 1) == 0;
                    float limit = xCoordinate ? detailed.getWidth() : detailed.getHeight();
                    if (Float.isNaN(value) || Float.isInfinite(value) ||
                            value < -2.0f || value > limit + 2.0f) {
                        throw new IllegalStateException("OCR box coordinate is out of range");
                    }
                }
                if (!validScore(line.getDetScore()) || !validScore(line.getRecScore())) {
                    throw new IllegalStateException("OCR score is out of range");
                }
            }
            // Explicitly exercise idempotent close; try-with-resources closes
            // the same engine once more at scope exit.
            ocr.close();
            ocr.close();
        }
        long elapsedMs = (System.nanoTime() - started) / 1_000_000L;

        System.out.println("image: " + imagePath);
        System.out.println("size: " + detailed.getWidth() + "x" + detailed.getHeight());
        System.out.println("lines: " + detailed.getLines().size());
        System.out.println("Java end-to-end elapsed: " + elapsedMs + " ms");
        for (OcrLine line : detailed.getLines()) {
            float[] box = line.getBox();
            System.out.printf("[%02d] %s%n", line.getIndex() + 1, line.getText());
            System.out.printf("      box: (%.1f, %.1f) (%.1f, %.1f) (%.1f, %.1f) (%.1f, %.1f)%n",
                    box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]);
            System.out.printf("      det=%.3f rec=%.3f%n", line.getDetScore(), line.getRecScore());
        }
    }

    private static boolean validScore(float score) {
        return !Float.isNaN(score) && !Float.isInfinite(score) && score >= 0.0f && score <= 1.0f;
    }
}
