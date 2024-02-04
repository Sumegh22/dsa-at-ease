package basicsofprogramming.assignment3.Loops;

public class ForLoopEx {

    public static void main(String[] args) {

        String s = "HelloWorld";
        for (int i = s.length() - 1; i >= 0; i--) {
            System.out.print(s.charAt(i));
        }
    }
}
