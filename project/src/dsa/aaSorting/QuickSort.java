package project.src.dsa.aaSorting;

public class QuickSort {

    public static void main(String[] args) {
        int[] arr = {9, 7, 5, 11, 2, 8, 12, 13, 2};

        System.out.print("Original array: ");
        for (int num : arr) {
            System.out.print(num + " ");
        }
        System.out.println();
        QuickSort qs = new QuickSort();
        qs.sortByQuickSort(arr);

        System.out.print("After sorting : ");
        for (int num : arr) {
            System.out.print(num + " ");


        }
    }

    void sortByQuickSort(int[] nums){
        int n = nums.length;
        sorting(nums, 0 , n-1);
    }

    void sorting(int[] arr, int low, int high){
        if(low<high){
            int pi = getPartitionIndex(arr, low, high);
            sorting(arr, low, pi-1);
            sorting(arr, pi+1, high);
        }
    }

    int getPartitionIndex(int[] arr, int low, int high){
        int pivot = low;
        int i = low;
        int j = high;

        while(i<j){
            /*  Move i to the right until we find an element greater than the pivot  */
            while (arr[i] <= arr[pivot] && i <= high - 1) {
                i++;
            }
            /*  Move j to the left until we find an element smaller than the pivot  */
            while (arr[j] > arr[pivot] && j >= low + 1) {
                j--;
            }
            /*  Swap elements at i and j if i is still less than j  */
            if (i < j) {
                swap(arr, i , j);
            }
        }

        // Pivot placed in correct position
        swap(arr, low, j);
        return  j;

    }

    static void swap(int[] arr, int a, int b) {
        int temp = arr[a];
        arr[a] = arr[b];
        arr[b] = temp;
    }
}
