package project.src.dsa.aaBasicConcepts.basicmath;

public class PrimeNumber {

    public boolean isPrime(int n) {
        if(n==1) return false;

        for(int i=2; i*i<=n; i++){
            if(n%i ==0){
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
        int n = 5;

        /* Creating an instance of
        Solution class */
        PrimeNumber sol = new PrimeNumber();

        /* Function call to find whether the
         given number is prime or not */
        boolean ans = sol.isPrime(n);

        if (ans) {
            System.out.println(n + " is a prime number.");
        } else {
            System.out.println(n + " is not a prime number.");
        }
    }
}
