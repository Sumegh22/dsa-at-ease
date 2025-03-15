package project.src.dsa.aaSorting;

public class SelectionSort {

    static void sortBySelectionSort(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int mini = i;
            for (int j = i + 1; j < n; j++) {
                if (nums[mini] > nums[j]) {
                    mini = j;
                }
            }
            swap(nums, i, mini);
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
        sortBySelectionSort(arr);

        System.out.print("After sorting : ");
        for (int num : arr) {
            System.out.print(num + " ");


        }
    }
}
