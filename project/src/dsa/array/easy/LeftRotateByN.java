package project.src.dsa.array.easy;


public class LeftRotateByN {

    // Uses 2 pointer approach where 1 pointer is at start and 2nd pointer is at end

    // Function to reverse the array between start and end

    private void reverseArray(int[] nums, int start, int end) {

        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }

    // Function to rotate the array to the left by k positions
    public void rotateArray(int[] nums, int k) {
        int n = nums.length; // Size of array
        k = k % n; // To avoid unnecessary rotations

        // Reverse the first k elements
        reverseArray(nums, 0, k - 1);

        // Reverse the last n-k elements
        reverseArray(nums, k, n - 1);

        // Reverse the entire array
        reverseArray(nums, 0, n - 1);
    }

    public static void printArray(int[] nums) {
        for (int val : nums) {
            System.out.print(val + " ");
        }
        System.out.println();
    }
    // Helper function to print the array

    public static void main(String[] args) {
        int[] nums = {1, 2, 3, 4, 5, 6};
        int k = 72;

        LeftRotateByN sol = new LeftRotateByN();

        System.out.println("Initial array: ");
        printArray(nums);

        // Create an instance of the Solution class


        /* Function call to rotate the
        array to the left by k places */
        sol.rotateArray(nums, k);

        System.out.println("Array after rotating elements by " + k + " places: ");
        printArray(nums);
    }
}

