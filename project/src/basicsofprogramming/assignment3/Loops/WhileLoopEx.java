package basicsofprogramming.assignment3.Loops;

public class WhileLoopEx {

    static void findAllCategories(String str){
        int i = 0;
        while(i<str.length()) {
            int found = str.indexOf("category:", i);
            if(found == -1){ break; }
            int start = found+9;
            int end = str.indexOf(" ", start);
            String subStr = str.substring(start, end);
            System.out.println(subStr);
            i = end + 1;
        }
    }

    public static void main(String[] args) {
        String input= "category:hell , category:yeah , category:hellYeah , category:apperal , category:cuisine ";
        findAllCategories(input);

    }


}
