package project.src.dsa.aaSorting;

import java.util.ArrayList;

public class MergeSort {

    public static void main(String[] args) {
        int[] arr = {7, 5, 9, 2, 8};

        System.out.print("Original array: ");
        for (int num : arr) {
            System.out.print(num + " ");
        }
        System.out.println();
        MergeSort ms = new MergeSort();
        ms.sortByMergeSort(arr);

        System.out.print("After sorting : ");
        for (int num : arr) {
            System.out.print(num + " ");


        }
    }

    void sortByMergeSort(int[] nums){
        int n = nums.length;
        sorting(nums, 0, n);
    }

    void sorting(int[] arr, int begin, int end) {
        if(begin>= end){
            return;
        }
        int mid = (begin+end) / 2;
        sorting(arr, begin, mid);
        sorting(arr, mid+1, end-1);
        merge(arr, begin, mid, end-1);
    }

    void merge(int[] arr, int begin, int mid, int end){
        int left = begin;
        int right = mid+1;
        ArrayList<Integer> list = new ArrayList<>();

        while(left<=mid && right<=end){
            if(arr[left]<= arr[right]){
                list.add(arr[left]);
                left++;
            } else{
                list.add(arr[right]);
                right++;
            }
        }
        while(left<=mid){
            list.add(arr[left]);
            left++;
        }
        while(right<=end){
            list.add(arr[right]);
            right++;
        }
        for(int i = begin; i<=end; i++){
            arr[i] = list.get(i-begin);
        }
    }

}
