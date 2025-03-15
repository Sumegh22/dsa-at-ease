package project.src.dsa.aaSorting;

public class InsertionSort {
    static void insertionSort(int[] nums){
        int n = nums.length;

        for(int i=0; i<n; i++){
            int j = i;
            while(j>0 && nums[j-1]>nums[j]){
                swap(nums, j-1, j);
                j--;
            }
        }
    }

    static void swap(int[] arr, int a, int b) {
        int temp = arr[a];
        arr[a] = arr[b];
        arr[b] = temp;
    }

    public static void main(String[] args) {
        int[] arr = {7, 5, 9, 2, 8};

        System.out.print("Original array: ");
        for (int num : arr) {
            System.out.print(num + " ");
        }
        System.out.println();
        insertionSort(arr);

        System.out.print("After sorting : ");
        for (int num : arr) {
            System.out.print(num + " ");


        }
    }
}
