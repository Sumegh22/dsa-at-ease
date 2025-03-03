package project.src.dsa.hashing;

import java.util.HashMap;
import java.util.Map;

public class MostFrequentElementInArray {

    /*
     We can use Map or HashArray for this problem
        1. Using map store element as key and its occurrence as value in map
        2. For HashArray, create hash array greater than length, increament the occurrence of a[i] everytime same element appears
     */

    public int mostFrequentElementByMap(int[] nums) {
        HashMap<Integer, Integer> frequencyMap = new HashMap<>();
        for(int n : nums){
            frequencyMap.put(n, frequencyMap.getOrDefault(n, 0)+ 1);
        }
        return frequencyMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    }

    public int mostFrequentElementByHashArray(int[] nums) {


        int n = nums.length;
        int maxFreq = 0;

        /* Variable to store element
        with maximum frequency */
        int maxEle = 0;

        // Visited array
        boolean[] visited = new boolean[n];

        // First loop
        for (int i = 0; i < n; i++) {
            // Skip second loop if already visited
            if (visited[i]) continue;

            /* Variable to store frequency
            of current element */
            int freq = 0;

            // Second loop
            for (int j = i; j < n; j++) {
                if (nums[i] == nums[j]) {
                    freq++;
                    visited[j] = true;
                }
            }

            /* Update variables if new element having
            highest frequency is found */
            if (freq > maxFreq) {
                maxFreq = freq;
                maxEle = nums[i];
            } else if (freq == maxFreq) {
                maxEle = Math.min(maxEle, nums[i]);
            }
        }

        // Return the result
        return maxEle;
    }

    public static void main(String[] args) {
        int[] nums = {4, 4, 5, 5, 6};

        /* Creating an instance of
        Solution class */
        MostFrequentElementInArray sol = new MostFrequentElementInArray();

        /* Function call to get the
        highest occurring element in array nums */
        int ans = sol.mostFrequentElementByMap(nums);

        System.out.println("The highest occurring element in the array is: " + ans);
    }

}
