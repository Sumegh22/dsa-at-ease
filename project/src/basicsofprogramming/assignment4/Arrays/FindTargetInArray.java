package basicsofprogramming.assignment4.Arrays;

public class FindTargetInArray {

    static int searchInArray(int[] array, int value){
         int ret = -1;
         for( int i =0; i<array.length; i++){
             if (array[i] == value ){
                 ret = i;
                 break;
             }
         }
        return ret;
    }

    public static void main(String[] args) {
        int[] array = {1,4,9,16,25,36,49,64,81,100, 1, 2};
        System.out.println("given element is found in array at index: " + searchInArray(array, 1));

    }

}
