package project.src.dsa.array.hard;

import java.util.Arrays;

public class MergeTwoSortedArrays {

    public void swapIfGreater(int[] nums1, int[] nums2, int a, int b) {
        if (nums1[a] > nums2[b]) {
            int temp = nums1[a];
            nums1[a] = nums2[b];
            nums2[b] = temp;
        }
    }

    public void mergeArrays1(int[] nums1, int m, int[] nums2, int n) {
        // This approach is based on 2 pointer approach
        int left = m - 1;
        int right = 0;

        while (left >= 0 && right < n) {
            if (nums1[left] < nums2[right]) {
                swapIfGreater(nums1, nums2, left, right);
                left--;
                right++;
            } else {
                break;
            }
        }
        Arrays.sort(nums1, 0, m);
        Arrays.sort(nums2);

        for (int i = m; i < m + n; i++) {
            nums1[i] = nums2[i - m];
        }
    }

    void mergeArrays2(int[] nums1, int m, int[] nums2, int n) {
        // this approach is based on shell sort algorithm

        int len = m + n;
        int gap = len / 2 + len % 2;// Adding remainder for ceil value;

        while (gap > 0) {
            int left = 0;
            int right = left + gap;

            while (right < len) {
                //ar1  and ar2
                if (left < m && right >= m) {
                    swapIfGreater(nums1, nums2, left, right - m);
                }
                //ar2 and ar2
                else if (left >= m) {
                    swapIfGreater(nums2, nums2, left - m, right - m);
                } else {
                    swapIfGreater(nums1, nums1, left, right);
                }
                left++;
                right++;
            }
            if (gap == 1) break;

            gap = (gap / 2) + (gap % 2);
        }

        for (int i = m; i < m + n; i++) {
            nums1[i] = nums2[i - m];
        }

    }


    public static void main(String[] args) {
        int[] nums1 = {-5, -2, 4, 5, 0, 0, 0};
        int[] nums2 = {-3, 1, 8};
        int m = 4, n = 3;

        // Create an instance of the Solution class
        MergeTwoSortedArrays sol = new MergeTwoSortedArrays();

        sol.mergeArrays2(nums1, m, nums2, n);

        // Output the merged arrays
        System.out.println("The merged arrays are:");
        System.out.print("nums1[] = ");
        for (int num : nums1) {
            System.out.print(num + " ");
        }

        System.out.println();
    }

}
