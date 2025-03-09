package project.src.dsa.aaBasicConcepts.basicmath;

public class Factorial {

    public int factorial(int n) {
        if(n<=1){
            return 1;
        }
        return n*factorial(n-1);
    }

    public static void main(String[] args) {
        Factorial sol = new Factorial();
        int n = 5;
        System.out.println("Factorial of " + n + " is: " + sol.factorial(n));
    }

}
