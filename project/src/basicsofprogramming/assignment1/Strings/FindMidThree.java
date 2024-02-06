package basicsofprogramming.assignment1.Strings;

public class FindMidThree {

    public static String findMidThree(String str){
        if(str.length() >3 ){
            int halfLen = str.length()/2;
            return str.substring(halfLen-1, halfLen+2);
        }
        return str;
    }

    /**
     Given a string of odd length, return the middle 3 characters from the string,
     so the string <b>"Monitor"</b> yields <b>"nit"</b>.
     If the string length is less than 3, return the string as is. <br> <br>

     <b>EXPECTATIONS:</b><br>
     middleThree("bunny") <b>---></b> "unn" <br>
     middleThree("peter") <b>---></b> "ete" <br>
     middleThree("Jamaica") <b>---></b>"mai" <br>
     */

    public static void main(String[] args) {
        System.out.println(findMidThree("jamaica"));

    }

}
