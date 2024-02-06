package project.src.logic.puzzle.src;

public class TwoSumOne {

	/**
	 * Given three ints, a b c, return true if it is possible to add two of 
	 * the ints to get the third. There should only be a single line of code in the method body.<br>
	 * <br> 
	 *
	 * <b>EXPECTATIONS:</b><br>
		twoSumOne(1, 2, 3)   <b>---></b> true <br>
		twoSumOne(3, 1, 2)    <b>---></b> true <br>
		twoSumOne(3, 2, 2) <b>---></b> false <br>
	 */
		public static boolean twoSumOne(int a, int b, int c) {
			return (a == b+c) || (b == a+c ) || (c == a+b);

		}

	//----------------------STARTING POINT OF PROGRAM. IGNORE BELOW --------------------//
		public static void main(String args[]){
			System.out.println(twoSumOne(16,2,18));
		}
			

}
