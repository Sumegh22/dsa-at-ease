package project.src.dsa.aaBasicConcepts.basicmath;

public class largestDigitInNumber {
    public int largestDigit(int n) {
        // Use a while loop and a max variable, Modulo by 10 to get the last digit, compare it with max and update max
        // And divide by 10 to update given number, return the max
        int max = 0;
        while (n > 0) {
            int digit = n % 10;
            if (digit > max) {
                max = digit;
            }
            n = n / 10;
        }
        return max;
    }

    public static void main(String[] args) {
        largestDigitInNumber sol = new largestDigitInNumber();
        int n = 123945;
        System.out.println("Largest digit in " + n + " is: " + sol.largestDigit(n));
    }

}
