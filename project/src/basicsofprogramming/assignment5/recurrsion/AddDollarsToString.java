package basicsofprogramming.assignment5.recurrsion;

public class AddDollarsToString {

    public static String allDollars(String input){
        if(input.length()<=1){
            return input;
        }
        return input.charAt(0)+"$"+allDollars(input.substring(1));
    }

    public static void main(String[] args) {
        System.out.println(allDollars("hello"));
    }
}
