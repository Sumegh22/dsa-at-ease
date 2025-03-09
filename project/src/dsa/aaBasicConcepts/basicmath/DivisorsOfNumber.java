package project.src.dsa.aaBasicConcepts.basicmath;

import java.util.ArrayList;

public class DivisorsOfNumber {
    public int[] divisors(int n) {
        ArrayList<Integer> list = new ArrayList<>();
        for(int i=1; i*i<=n; i++){
            if(n%i ==0){
                list.add(i);
                if(n/i !=i){
                    list.add(n/i);
                }
            }
        }
        return list.stream().sorted().mapToInt(i -> i).toArray();
    }

    public static void main(String[] args) {
        int n = 6;

        /* Creating an instance of  Solution class */
        DivisorsOfNumber sol = new DivisorsOfNumber();

        /* Function call to find all divisors of n */
        int[] ans = sol.divisors(n);

        System.out.print("The divisors of " + n + " are: ");
        for (int i = 0; i < ans.length; i++) {
            System.out.print(ans[i] + " ");
        }
    }
}
