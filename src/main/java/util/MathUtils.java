package util;

/** כלים בסיסיים ל-LDA: פתרון מערכות לינאריות, לוג בטוח ועוד. */
public final class MathUtils {
    private MathUtils() {}

    /** פתרון Ax = b באלימינציה גאוסית עם Pivoting חלקי. לא מחשבים אינברס. */
    public static double[] solveLinearSystem(double[][] A, double[] b) {
        int n = A.length;
        if (A[0].length != n) throw new IllegalArgumentException("A must be square");
        if (b.length != n) throw new IllegalArgumentException("b length mismatch");

        // מטריצה מאוגמנטת [A|b]
        double[][] M = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, M[i], 0, n);
            M[i][n] = b[i];
        }

        // אלימינציה עם Pivoting חלקי + נרמול שורה
        for (int col = 0; col < n; col++) {
            // בחירת פיבוט
            int pivot = col;
            double max = Math.abs(M[col][col]);
            for (int r = col + 1; r < n; r++) {
                double val = Math.abs(M[r][col]);
                if (val > max) { max = val; pivot = r; }
            }
            if (Math.abs(M[pivot][col]) < 1e-12)
                throw new IllegalArgumentException("Matrix is singular or ill-conditioned");

            // החלפת שורות אם צריך
            if (pivot != col) {
                double[] tmp = M[pivot]; M[pivot] = M[col]; M[col] = tmp;
            }

            // נרמול שורת הפיבוט ל-1 בעמודת col
            double piv = M[col][col];
            for (int j = col; j <= n; j++) M[col][j] /= piv;

            // איפוס שאר העמודה
            for (int r = 0; r < n; r++) {
                if (r == col) continue;
                double factor = M[r][col];
                if (factor == 0) continue;
                for (int j = col; j <= n; j++) {
                    M[r][j] -= factor * M[col][j];
                }
            }
        }

        // הפתרון נמצא בעמודה האחרונה
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = M[i][n];
        return x;
    }

    /** רגולריזציה: הוספת λ לאלכסון (ליציבות Σ). */
    public static void addToDiagonal(double[][] A, double lambda) {
        for (int i = 0; i < A.length; i++) A[i][i] += lambda;
    }

    /** מכפלת נקודה. */
    public static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    /** לוג בטוח למניעת NaN. */
    public static double safeLog(double x) {
        return Math.log(Math.max(x, 1e-12));
    }
}
