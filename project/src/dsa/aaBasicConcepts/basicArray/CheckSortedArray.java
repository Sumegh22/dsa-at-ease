package project.src.dsa.aaBasicConcepts.basicArray;

public class CheckSortedArray {
    // Function to check if an array is sorted
    public boolean arraySortedOrNot(int[] arr, int n) {
        // Iterate through the array
        for (int i = 0; i < n - 1; i++) {
            // Compare each element with the next one
            if (arr[i] > arr[i + 1]) {

                /* If any element is greater than the next one, the array is not sorted */
                return false;
            }
        }
        return true; // If no such pair is found, array is sorted
    }

    public static void main(String[] args) {
        // Creating an instance of solution class
        CheckSortedArray solution = new CheckSortedArray();

        int[] arr = {1, 2, 3, 4, 5};
        int n = arr.length;

        // Function call to check if the array is sorted
        boolean sorted = solution.arraySortedOrNot(arr, n);

        if (sorted) {
            System.out.println("Array is sorted.");
        } else {
            System.out.println("Array is not sorted.");
        }
    }
}
