package project.src.dsa.aaBasicConcepts.basicmath;

public class ReverseNumber {

    // Reverse number logic can be used to test Palindrome too

    public int reverseNumber(int n) {
        int rev = 0;
        while(n>0){
            rev= rev*10+ n%10;
            n/=10;
        }
        return rev;
    }

    public static void main(String[] args) {
        ReverseNumber sol = new ReverseNumber();
        int n = 12321;
        System.out.println("Number of digits in " + n + " is: " + sol.reverseNumber(n));
    }
}
