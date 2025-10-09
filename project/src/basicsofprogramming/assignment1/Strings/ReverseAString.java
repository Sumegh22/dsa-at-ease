package basicsofprogramming.assignment1.Strings;


public class ReverseAString {

    public static String reverse1(String str){
      StringBuilder sb = new StringBuilder(str);
        return sb.reverse().toString();
    }

      public static String reverse2(String s){
        int left = 0;
        int right = s.length()-1;

        while(left<right) {
          char c = s.charAt(left);
          s.replace(s.charAt(left), s.charAt(right));
          s.replace(s.charAt(right), c);
          left++; right--;
        }          
        return s;
    }

      public static String reverseByRecursion(String str){
        if(str.length()<=1 || str == null) return str;
        
        return reverseByRecursion(str.substring(1))+str.charAt(0);
    }

    /**

    The mentioned code can be used to reverse any given String, Three possible apporaches are mentioned in this class
    
    */
  
    public static void main(String[] args) {
        System.out.println(reverseByRecursion("jamaica"));

    }

}
