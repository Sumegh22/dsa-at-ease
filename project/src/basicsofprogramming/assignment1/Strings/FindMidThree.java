package basicsofprogramming.assignment1.Strings;

public class FindMidThree {

    public static String findMidThree(String str){
        if(str.length() >3 ){
            int halfLen = str.length()/2;
            return str.substring(halfLen-1, halfLen+2);
        }
        return str;
    }

    public static void main(String[] args) {
        System.out.println(findMidThree("jamaica"));

    }

}
