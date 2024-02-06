package project.src.basicsofprogramming.assignment2.conditinals;

public class SumUniqueNums {

    static int addUniqueNums(int a, int b, int c){
      int sum = 0;
      if(a!=b && a!=c){ sum+=b+c;}
        if(b!=c && b!=a){ sum+=a+c;}
        if(c!=a && c!=b){ sum+=a+b;}
        return sum;
    }

    public static void main(String[] args) {
        System.out.println(addUniqueNums(1,2,2));
    }
}
