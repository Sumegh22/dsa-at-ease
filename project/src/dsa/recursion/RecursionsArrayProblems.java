package project.src.dsa.recursion;

import java.util.ArrayList;

public class RecursionsArrayProblems {
    public static void main(String[] args) {
        int[] arr = {1,2,4,8,9, 0, 8};
        int[] unsorted = {4,5,8,1};

        System.out.println(findAllOccurrences(arr, 8, 0, new ArrayList<Integer>()));
    }

    //----------------------------------------------------------//

    static boolean checkSorted(int[] arr, int index){
        if(index == arr.length) return  true;
        return checkSorted(arr, index+1);
    }

    //----------------------------------------------------------//

    static int linearSearchRecursion(int[] arr, int target, int index){
        if(index == arr.length) return -1;
        if(arr[index] == target) return index;

        return linearSearchRecursion(arr, target, index+1);

    }
    //----------------------------------------------------------//

    static ArrayList<Integer> findAllOccurrences(int[] arr, int target, int index, ArrayList<Integer> list){
        if(index == arr.length) return list;
        if(arr[index] == target) list.add(index);

        return findAllOccurrences(arr, target, index+1, list);

    }
}
