/** Immutable detailed OCR line in source-image coordinates. */
public final class OcrLine {
    private final int index;
    private final String text;
    private final float[] box;
    private final float detScore;
    private final float recScore;

    public OcrLine(int index, String text, float[] box, float detScore, float recScore) {
        if (text == null) {
            throw new NullPointerException("text");
        }
        if (box == null || box.length != 8) {
            throw new IllegalArgumentException("box must contain 8 coordinates");
        }
        this.index = index;
        this.text = text;
        this.box = box.clone();
        this.detScore = detScore;
        this.recScore = recScore;
    }

    public int getIndex() { return index; }
    public String getText() { return text; }
    /** Returns x1,y1,x2,y2,x3,y3,x4,y4 as a defensive copy. */
    public float[] getBox() { return box.clone(); }
    public float getDetScore() { return detScore; }
    public float getRecScore() { return recScore; }
}
