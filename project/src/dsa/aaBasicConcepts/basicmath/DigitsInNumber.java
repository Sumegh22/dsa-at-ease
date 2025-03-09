package project.src.dsa.aaBasicConcepts.basicmath;

public class DigitsInNumber {
    // Use a while loop and a count variable, divide by 10 in each pass and increment count var
    //
    public int countDigits(int n) {
        int count = 0;
        while (n > 0) {
            n = n / 10;
            count++;
        }
        return count;
    }

    public static void main(String[] args) {
        DigitsInNumber sol = new DigitsInNumber();
        int n = 12345;
        System.out.println("Number of digits in " + n + " is: " + sol.countDigits(n));
    }

}
