package project.src.dsa.aaSorting;

public class BubbleSort {
    static void sortByBubbleSort(int[] nums){
        int n = nums.length;

        for(int i=0; i<n; i++){
            for(int j=0; j<n-1-i; j++){
                if(nums[j]>nums[j+1]){
                    swap(nums, j+1 , j);
                }
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
        sortByBubbleSort(arr);

        System.out.print("After sorting : ");
        for (int num : arr) {
            System.out.print(num + " ");


        }
    }
}
