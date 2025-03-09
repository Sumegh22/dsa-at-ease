package project.src.dsa.aaBasicConcepts.basicmath;

public class OddDigitsInNumber {
    public int countOddDigits(int n) {
        // Use a while loop and a count variable, Modulo by 10 to get the last digit, check if it is odd increment the counter
        // And divide by 10 to update given number, return the count
        int count = 0;
        while (n > 0) {
            int digit = n % 10;
            if (digit % 2 != 0) {
                count++;
            }
            n = n / 10;
        }
        return count;
    }

    public static void main(String[] args) {
        OddDigitsInNumber sol = new OddDigitsInNumber();
        int n = 12345;
        System.out.println("Number of odd digits in " + n + " is: " + sol.countOddDigits(n));
    }

}
