package project.src.dsa.aaBasicConcepts.basicmath;

public class ArmStrongNumber {

    public boolean isArmstrong(int n) {
        int copy= n;
        int len=0;
        int validator = n;
        double sum=0;

//        while(copy>0){
//            len++;
//            copy/=10;
//        }
        len = (int) Math.log10(n) + 1;

        while(n>0){
            sum+= Math.pow(n%10, len);
            n/=10;
        }

        return sum==validator;

    }

    public static void main(String[] args) {
        ArmStrongNumber sol = new ArmStrongNumber();
        int n = 153;
        System.out.println("Is " + n + " an Armstrong number? " + sol.isArmstrong(n));
    }

}
