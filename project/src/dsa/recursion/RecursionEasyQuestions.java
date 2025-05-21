package project.src.dsa.recursion;

public class RecursionEasyQuestions {
    public static void main(String[] args) {
        int n = 1024;
        int ans = reduceToZero(n);
        System.out.println("ans is = "+ans);

    }

//---------------------------------------------------------------//
    static int sumOfDigit(int num){
        if(num==0) return 0;

        return num%10+ sumOfDigit(num/10);
    }

//---------------------------------------------------------------//

    static int prodOfDigit(int num){
        if(num%10 == num) return num;
        return (num%10) * prodOfDigit(num/10);
    }
//---------------------------------------------------------------//

    static void print1toN(int n){
        if(n <1) return ;
        print1toN(n-1);
        System.out.println(n);
    }

//---------------------------------------------------------------//

    static int reverseANumber(int num) {
        int numOfDigits = (int) Math.log10(num)+1;
        return reverse(num, numOfDigits);
    }

    private static int reverse(int num, int numOfDigits){
        if(num%10 == num) return num;
        int ld = num%10;

        return ld * (int) Math.pow(10, numOfDigits-1)
                + reverse((num / 10), numOfDigits - 1);

    }
//---------------------------------------------------------------//

    static boolean palindromeNumber(int n){
        return n == reverseANumber(n);
    }
//---------------------------------------------------------------//

    static int countZerosInNumber(int num, int count){
        if(num ==0) return count;
        if(num%10 == 0 ) return countZerosInNumber(num/10, count+1);
        else return countZerosInNumber(num/10, count);
    }
//---------------------------------------------------------------//
/* ----- Leetcode question Steps to reduce a number to 0 ------*/

    static int reduceToZero(int num){
        return reduce(num, 0);
    }

    private static int reduce(int num, int steps){
        if (num == 0) {
            return  steps;
        }
        if(num%2 == 0) {
            return reduce(num/2, steps+1);
        }
        return reduce(num-1, steps+1);
    }

//---------------------------------------------------------------//

    static String reverseStringByRecursion(String s){
        if(s.length() == 0) return"";
        return reverseStringByRecursion(s.substring(1))+s.charAt(0);
    }
//---------------------------------------------------------------//

}
