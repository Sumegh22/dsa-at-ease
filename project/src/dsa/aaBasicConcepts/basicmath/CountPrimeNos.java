package project.src.dsa.aaBasicConcepts.basicmath;

public class CountPrimeNos {

    // Function to find whether the number is prime or not
    private boolean isPrime(int n) {
        // Variable to store the count of divisors of n
        int count = 0;

        // Loop from 1 to square root of n
        for (int i = 1; i <= Math.sqrt(n); ++i) {
            // Check if i is a divisor
            if (n % i == 0) {
                // Increment count
                count = count + 1;

                /* Check if counterpart divisor is
                different from original divisor */
                if (n / i != i) {
                    // Increment count
                    count = count + 1;
                }
            }
        }

        // If count is 2, n is prime
        return count == 2;
    }

    // Function to find count of primes till n
    public int primeUptoN(int n) {
        // Variable to store count
        int count = 0;

        // Iterate from 1 to n
        for (int i = 2; i <= n; i++) {
            // Check if i is prime
            if (isPrime(i)) count++;
        }

        // Return the count
        return count;
    }

    public static void main(String[] args) {
        int n = 6;

        // Creating an instance of Solution class
        CountPrimeNos sol = new CountPrimeNos();

        /* Function call to get count
        of all primes till n */
        int ans = sol.primeUptoN(n);

        System.out.println("The count of primes till " + n + " is: " + ans);
    }

}
