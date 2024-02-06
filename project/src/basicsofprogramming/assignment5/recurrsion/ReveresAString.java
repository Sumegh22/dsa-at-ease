package project.src.basicsofprogramming.assignment5.recurrsion;

public class ReveresAString {

    static String reverse(String str){
        int len = str.length();
        if (len<=1){
            return  str;
        }
        return reverse(str.substring(1))+str.charAt(0);
    }

    public static void main(String[] args) {
        System.out.println(reverse("sumegh"));
    }
}
