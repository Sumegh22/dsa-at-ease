package project.src.dsa.array.easy;

import java.util.Arrays;
import java.util.Comparator;

public class SecondLargest {
    public static void main(String[] args) {
        int[] input = {12, 39, 1, 10, 34, 131};

        int secondLargest = secondLargestElementByStreams(input);
        if (secondLargest == Integer.MIN_VALUE) {
            System.out.println("No second largest element found");
        } else {
            System.out.println("Second Largest Element: " + secondLargest);
        }
    }

    public static int secondLargestElementByStreams(int[] nums) {
        return Arrays.stream(nums)
                .boxed()  // Convert int to Integer for sorting & distinct operations
                .distinct() // Remove duplicates
                .sorted(Comparator.reverseOrder()) // Sort in descending order
                .skip(1) // Skip the largest element
                .findFirst() // Get the second largest element
                .orElse(-1); // If no second largest exists, return -1
    }

    public static int secondLargestElement(int[] nums) {
        int largest = nums[0];
        int sl = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > largest) {
                sl = largest;
                largest = nums[i];
            } else if (nums[i] > sl && nums[i] < largest) {
                sl = nums[i];
            }
        }
        return sl;

    }

}

