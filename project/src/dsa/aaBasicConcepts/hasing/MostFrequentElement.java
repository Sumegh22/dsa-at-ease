package project.src.dsa.aaBasicConcepts.hasing;

public class MostFrequentElement {
    // Using streams and lib methods to solve the problem
    public int mostFrequentElement(int[] nums) {
        Map<Integer, Long> frequencyMap;
        if (nums == null) {
            return 0;
        }
        frequencyMap = Arrays.stream(nums)
                .boxed()
                .collect(Collectors.groupingBy(Integer::intValue, Collectors.counting()));

        return frequencyMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    }

    public static void main(String[] args) {
        int[] nums = {4, 4, 5, 5, 6, 7};

        /* Creating an instance of
        Solution class */
        MostFrequentElement sol = new MostFrequentElement();

        /* Function call to get the second
        highest occurring element in array */
        int ans = sol.mostFrequentElement(nums);

        System.out.println("The highest occurring element in the array is: " + ans);
    }
}
