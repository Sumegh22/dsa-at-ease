package project.src.basicsofprogramming.assignment2.conditinals;

public class FizzBuzz {

    /**
     Given an int n, return the string form of the number followed by "!".
     So if the int is for example 13 this method should return "13!".
     However the catch is that if the number is divisible by 3 the method should return "Fizz!"
     instead of the number, and if the number is divisible by 5 it should return "Buzz!",
     and if divisible by both 3 and 5, use "FizzBuzz!". You’ll need to use the % "mod"
     for computing the remainder after division, so 23 % 10 yields 3.

     <br>
     <br>

     * <b>EXPECTATIONS:</b><br>
     fizzyWizzy(1)   <b>---></b> "1!" <br>
     fizzyWizzy(2)    <b>---></b> "2!" <br>
     fizzyWizzy(3) <b>---></b> "Fizz!" <br>
     */
//    public static String fizzyWizzy(int n) {
//        if (n%3==0 ) {
//            return n % 5 == 0 ? "FizzBuzz!" : "Fizz!";
//        }
//        if (n%5==0){
//            return "Buzz!";
//        }
//        return n + "!";
//    }

    public static String fizzyWizzy(int n) {
        boolean fizz = n%3 ==0;
        boolean buzz = n%5==0;
        if (fizz && buzz){return "FizzBuzz!"; }
        else if (fizz) { return "Fizz!"; }
        else if (buzz) { return "Buzz!"; }

        return n+"!";
    }



    public static void main(String[] args) {
        System.out.println(fizzyWizzy(24));
    }
}
