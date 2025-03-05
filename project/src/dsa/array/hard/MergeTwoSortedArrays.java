package project.src.dsa.array.hard;

import java.util.Arrays;

public class MergeTwoSortedArrays {

    public void mergeArrays1(int[] nums1, int m, int[] nums2, int n) {
        int left = m - 1;
        int right = 0;

        while (left >= 0 && right < n) {
            if (nums1[left] < nums2[right]) {
                int temp = nums1[left];
                nums1[left] = nums2[right];
                nums2[right] = temp;
                left--;
                right++;
            } else {
                break;
            }
        }
        Arrays.sort(nums1, 0 , m);
        Arrays.sort(nums2);

        for (int i = m; i < m + n; i++) {
            nums1[i] = nums2[i - m];
        }
    }

    void mergeArrays2(){

    }


    public static void main(String[] args) {
        int[] nums1 = {-5, -2, 4, 5, 0, 0, 0};
        int[] nums2 = {-3, 1, 8};
        int m = 4, n = 3;

        // Create an instance of the Solution class
        MergeTwoSortedArrays sol = new MergeTwoSortedArrays();

        sol.mergeArrays1(nums1, m, nums2, n);

        // Output the merged arrays
        System.out.println("The merged arrays are:");
        System.out.print("nums1[] = ");
        for (int num : nums1) {
            System.out.print(num + " ");
        }

        System.out.println();
    }

}
