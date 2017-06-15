import java.util.Random;
import java.util.Scanner; 
import java.io.*;
public class  AnomalyDetection
{
  public int row=999,col=11;  //Rows and Column for training matrix
  public int crow=100,ccol=col;//Rows and Column for crossvalidiation matrix
  public int trow=4,tcol=col;//Rows and Column for test matrix 

  double m[][];//holds cofactors of determinants
 
 
  public void start()
  {
    int i=0,j=0,k=1;
    double mat[][]=new double[row][col]; //training set
    double matc[][]=new double[crow][ccol];//cross validiation set
    double matt[][]=new double[trow][tcol];//test set

    int yc[] = new int[crow];//traget (type) of coresponting crossvalidiation set
    int yt[] = new int[trow];//traget (type) of coresponting test set


     try
     { 
 
        BufferedReader ibr  =  new BufferedReader(new InputStreamReader(System.in));
        //System.out.print("\nEnter The File Name:");
        //String FILE_Train=ibr.readLine();
        //FILE_CROSS = FILE_CROSS+".txt";
       
/*Input training set*/

        System.out.println("Training Set ");

	String FILE_Train="Xtrain.txt";
        FileReader  fr = new FileReader(FILE_Train);
        BufferedReader br =  new BufferedReader(fr);
           
        Scanner s = null;
        String Data;

        for(i=0;i<row;i++)
        {
           if( (Data = br.readLine()) != null)
              s = new Scanner(Data);
              for(j=0;j<col;j++)
              {
                   if (s.hasNextDouble())
                      mat[i][j] =(float) s.nextDouble(); 
		   
                   //System.out.print(" "+mat[i][j]);         
              }
	      //System.out.println(); 
        }

/*Input cross validiation set*/

        System.out.println("Cross Validiation set");

	String FILE_Cross="Xcross.txt";
        FileReader  frc = new FileReader(FILE_Cross);
        BufferedReader brc =  new BufferedReader(frc);
           
        Scanner sc = null;
        String Datac;

        for(i=0;i<crow;i++)
        {
           if( (Datac = brc.readLine()) != null)
              sc = new Scanner(Datac);
              for(j=0;j<ccol;j++)
              {
                   if (sc.hasNextDouble())
                      matc[i][j] =(float) sc.nextDouble(); 
		   
                   //System.out.print(" "+matc[i][j]);         
              }
	      if (sc.hasNextDouble())
		yc[i] =(int) sc.nextDouble();
		
	      //System.out.print(" -> "+yc[i]+"\n"); 
        }

/*Input Test set*/

        System.out.println("Test set");

	String FILE_Test="Test.txt";
        FileReader  frt = new FileReader(FILE_Test);
        BufferedReader brt =  new BufferedReader(frt);
           
        Scanner st = null;
        String Datat;

        for(i=0;i<trow;i++)
        {
           if( (Datat = brt.readLine()) != null)
              st = new Scanner(Datat);
              for(j=0;j<tcol;j++)
              {
                   if (st.hasNextDouble())
                      matt[i][j] =(float) st.nextDouble(); 
		   
                   //System.out.print(" "+matt[i][j]);         
              }
	      if (st.hasNextDouble())
		yt[i] =(int) st.nextDouble();
		
	      //System.out.print(" -> "+yt[i]+"\n"); 
        }

    }
    catch(Exception e)
    {
       System.out.println("Error "+e);
       System.exit(0); 
    } 

    double mu[]=new double[col]; 
    mu= mean(mat);// finding mu for training set features
    //for(i=0;i<col;i++){System.out.println(" mu : "+mu[i]);}//printing mu

    double var[]=new double[col]; 
    var= sigma2(mat,mu);// finding sigma2 for training set features
    //for(i=0;i<col;i++){System.out.println(" Var: "+var[i]);}//printing varience

   
    double p[]=new double[crow];
    p = multivarate_gaussian(matc,mu,var);
    //for(i=0;i<crow;i++){System.out.println(" "+p[i]);}//printing pval
    //for(i=0;i<crow;i++){System.out.println(" "+yc[i]);}//printing yval
       
    //System.out.println("The epcilon calculation begins"); 
    double epsilon = select_threshold(yc,p);//finding epsilon
    System.out.println("The epcilon is: "+epsilon);

  }


  /*Mean Calculation*/
  public double[] mean(double matm[][])
  {
    //System.out.println("Mean Calculation");

    int i=0,j=0,k=1;         

    double mu[]=new double[col];
    double temp[]=new double[col];
               
    for(i=0;i<col;i++){temp[i]=0.0;}//initilization of temp to zero

    for(i=0;i<col;i++)
    {
     for(j=0;j<row;j++)
     {
       //System.out.print(" "+mat[j][i]);  
       temp[i] = temp[i] + matm[j][i];
     }
     temp[i] = temp[i]/((double)row); 
     //System.out.println(" >mu"+i+": "+temp[i]); 
    }   

    for(i=0;i<col;i++)
    {
      mu[i]=temp[i];
      //System.out.println(" "+mu[i]);  
    }//copy tem to mu

    return mu;

  }


  /*Varience Calculation*/
  public double[] sigma2(double matv[][],double mv[])
  {
    //System.out.println("Varience Calculation");

    int i=0,j=0,k=1;

    double sigma2[]=new double[col];
    double temp[]=new double[col]; 
               
    for(i=0;i<col;i++){temp[i]=0.0;}//initilization of temp to zero

    for(i=0;i<col;i++)
    {
     for(j=0;j<row;j++)
     {
       //System.out.print(" "+mat[j][i]); 
       temp[i] = temp[i] + (matv[j][i] - mv[i])*(matv[j][i] - mv[i]);
     }
     temp[i] = temp[i]/((double)row);//for proper use of sigma2 do not use -1 
     //System.out.println(" >sigma2"+i+": "+temp[i]);       
    }   

    for(i=0;i<col;i++)
    {
      sigma2[i]=temp[i];
      //System.out.println(" "+sigma2[i]);  
    }//copy tem to sigma2

    return  sigma2;
  }

  /*Computing multivariate gaussian*/
  public double[] multivarate_gaussian(double matg[][],double mg[],double vg[])
  {
    System.out.println("Computing multivariate gaussian");
   //p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2))

    int i,j;

    double covar[][]=new double[col][col];
    covar = sigmacovar(vg);

    /* printing covarience matrix
    System.out.println("Covarience matrix");  
    for(i=0;i<col;i++)
    {
     for(j=0;j<col;j++)
     {
       System.out.print(" "+covar[i][j]);
     }
     System.out.println();
    }*/

    double x_mean[][]=new double[crow][ccol];
    x_mean = x_mean(matg,mg);

    /* printing x - mean
    System.out.println("Covarience matrix minus mean value"); 
    for(i=0;i<crow;i++)
    {
     for(j=0;j<ccol;j++)
     {
       System.out.printf(" %.3f",x_mean[i][j]);
     }
     System.out.println();
    } */

    /*calculating determinants of covarience matrix*/
    double detsigma2 = det_sigma(covar,col);  
    //System.out.println("det :"+detsigma2);

    /*computing inverse of covarience matrix*/
    double matInv[][]=new double[col][col];
    for(i= 0 ; i< col; i++)
    {
      for(j = 0 ; j<col; j++)
      {
       matInv[i][j] = 0;
      }
      matInv[i][i] = 1;
    }
    mat_Inverse(col,col,covar,matInv);
    //System.out.println("Inverse of Covarience matrix is");
    //display(ccol,ccol,matInv);

    /*multiplying matrix X-mu with Inverse of covarience matrix*/
    double mul_res[][]=new double[crow][ccol];
    mul_res = Mat_Mul(x_mean,crow,col,matInv,col,col);
    //System.out.println("cross set matric  multiplication");
    //display(crow,col,mul_res);


   /*multiplying matrix X-mu with earlier result by element*/
    double mul_res_elt[][]=new double[crow][col];
    mul_res_elt = mat_Mul_elt(mul_res,crow,col,x_mean,crow,col);
    //System.out.println("next  multiplication");
    //display(crow,col,mul_res_elt);

    double pvalmvg[]=new double[crow];
    pvalmvg = sumrow_mat(crow,col,mul_res_elt);

    for(i =0; i<crow; i++)
    {
      pvalmvg[i]=Math.exp(-0.5*pvalmvg[i]);
      //System.out.println(" "+pvalmvg[i]); 
    }

    //System.out.println(" :"+Math.pow(detsigma2,-0.5));
    for(i =0; i<crow; i++)
    { 
      pvalmvg[i]=Math.pow(detsigma2,-0.5)*pvalmvg[i];
      //System.out.println(" "+pvalmvg[i]); 
    }

    double twoPI = (double)2.0 * 3.1416;
    double colbyTwo = (double)(-col/2.0);
    //System.out.println(" :"+Math.pow(twoPI,colbyTwo));

    for(i =0; i<crow; i++)
    { 
      pvalmvg[i]=Math.pow(twoPI,colbyTwo)*pvalmvg[i];
      //System.out.println(" "+pvalmvg[i]); 
    }

    return  pvalmvg;

  }

  /*Computing covarience matrix*/
  public double[][] sigmacovar(double varmg[])
  {
    //System.out.println("Computing covarience matrix");

    int i=0,j=0;
    double covarmg[][]=new double[col][col];

    for(i=0;i<col;i++)
    {
     for(j=0;j<col;j++)
     {
       if(i==j)
         covarmg[i][j]=varmg[i];
       else
         covarmg[i][j]=0.0;
     }  
    }

    return covarmg;

  }

  /*Coumuting x - mean*/
  public double[][] x_mean(double matxm[][],double mean[])
  {
    //System.out.println("Computing x - mean");

    int i=0,j=0,k=1;
    double x_m[][]=new double[crow][col];
               
    for(i=0;i<crow;i++)
    {
     for(j=0;j<col;j++)
     {
       x_m[i][j] = matxm[i][j] - mean[j];

       //System.out.print(" "+x_m[i][j]); 
     }
     //System.out.println(); 
    }
    return x_m;
   }

  /* find determinant of a square matrix */
  public double det_sigma(double A[][],int N)
  {

    double det=0.0;
    double res;


    if(N == 1)
      res = A[0][0];
    else if (N == 2)
    {
      res = A[0][0]*A[1][1] - A[1][0]*A[0][1];
    }
    else
    {
      res=0;

      for(int j1=0;j1<N;j1++)
      {
        m = new double[N-1][];

        for(int k=0;k<(N-1);k++)
          m[k] = new double[N-1];
					
        for(int i=1;i<N;i++)
        {
          int j2=0;
          for(int j=0;j<N;j++)
          {
            if(j == j1)
              continue;

            m[i-1][j2] = A[i][j];
            j2++;
          }
        }
        res += Math.pow(-1.0,1.0+j1+1.0)* A[0][j1] * det_sigma(m,N-1);
       }
     }
     return res;
   }


   /* This function find inverse of matrix */
   void mat_Inverse(int row1, int col1, double matin[][],double matin1[][])
   {
     //System.out.println(" Computing inverse ");

     boolean singular = false;

     int i, r, c;

     for(r = 0;( r < row1)&& !singular;  r++)
     {
       if((int)matin[r][r]!=0 )  /* Diagonal element is not zero */
       {
        for(c = 0; c < col1; c++)
        {
         if( c == r)
         {

           /* Make all the elements above and below the current principal
           diagonal element zero */

           double ratio =  matin[r][r];
           for( i = 0; i < col1; i++)
           {
             matin[r][i] /= ratio ;
             matin1[r][i] /= ratio;
           }
         }
         else
         {
           double ratio = matin[c][r] / matin[r][r];
           for( i = 0; i < col1;  i++)
           {
             matin[c][i] -= ratio * matin[r][i];
             matin1[c][i] -= ratio * matin1[r][i];
           }
         }
        } 
       }
       else
       {
         /* If principal diagonal element is zero */
         singular = true;

         for(c = (r+1); (c < col1) && singular; ++c)

         if((int)matin[c][r]!=0)
         {
            singular = false;
            /* Find non zero elements in the same column */
            swap(r,c,col1, matin, matin1);
            --r;
         }
       }
     }
   }


  /* This method is a part of Inverse function */
  void swap( int row1,int row2, int col, double mats[][],double mats1[][])
  {
   for(int i = 0; i < col; i++)
   {
     double   temp = mats[row1][i];
     mats[row1][i] = mats[row2][i];
     mats[row2][i] = temp;

     temp = mats1[row1][i];
     mats1[row1][i] = mats1[row2][i];
     mats1[row2][i] = temp;
   }
  }

                                                       
  /* This method find the multiplication of two Matrix */
  public double[][] Mat_Mul(double mat1[][],int row1,int col1,double mat2[][],int row2,int col2)
  {
    int i, j, k;
    double mat_mul_res[][] = new double[row1][col2]; 

    if(col1 == row2)
    {
      for(i =0; i<row1; i++)
        for(j=0; j<col2; j++)
        {
          mat_mul_res[i][j] = 0;

          for(k = 0; k < col1; k ++)
          {
            mat_mul_res[i][j] += mat1[i][k] * mat2[k][j];
          }
        }
    }
    else
     System.out.printf("\n Multiplication is not possible");

    return mat_mul_res;
   }

   /* This method find the multiplication of two Matrix by elements */
   public double[][] mat_Mul_elt(double mat1[][],int row1,int col1,double mat2[][],int row2,int col2)
   {
     int i, j, k;
     double mat_res_elt1[][]=new double[row1][col1];

     if(row1 == row2 && col1==col2)
     {
       for(i =0; i<row1; i++)
         for(j=0; j<col1; j++)
           mat_res_elt1[i][j] = mat1[i][j] * mat2[i][j];
     }
     else
       System.out.printf("\n Elements Multiplication is not possible");

     return mat_res_elt1;
   }
   /*computing summation of rows of an matrix*/
   public double[]  sumrow_mat(int row,int col,double mar_res1[][])
   {
     int i, j, k;
     double mat_res2[]=new double[row];

     for(i =0; i<row; i++)
       for(j=0; j<col; j++)
         mat_res2[i] += mar_res1[i][j];

     //for(i =0; i<row; i++)
      // System.out.println(" "+mat_res2[i]);

     return mat_res2;
   }

   void display( int row, int col, double mat1[][])
   {
      int i=0,j=0;
      System.out.println();
      /* Output of inverse Matrix */
      for( i = 0; i < row; i++)
      {
        for( j = 0; j < col; j++)
        {
          System.out.printf(" %.3f",mat1[i][j]);
        }
        System.out.println();
      }
   }


   double select_threshold(int yval[],double pval[])
   {

     System.out.println("Finding Epsilon");

     double bst_ep = 0.0;

     double bstF1  = 0.0;

     double F1 = 0.0;

     double epmx = max(pval);
     double epmn = min(pval);

     System.out.println("Max pval:"+max(pval));
     System.out.println("Min pval:"+min(pval));

     double stepsize = (epmx-epmn)/1000.0;

     double cr_ep=0.0;

     for(cr_ep = epmn; cr_ep <= epmx; cr_ep = cr_ep+stepsize)
     {
       int cvprediction[] = new int[crow];

       cvprediction = cv_prediction(pval,cr_ep);
       //for(int i=0;i<crow;i++){System.out.println(yval[i]+" : "+ cvprediction[i]);}

       int tp=0,fp=0,fn=0;

       tp = count_tp(cvprediction,yval);//System.out.println("tp:"+tp);
       fp = count_fp(cvprediction,yval);//System.out.println("fp:"+fp);
       fn = count_fn(cvprediction,yval);//System.out.println("fn:"+fn);
       
       if(tp==0) 
        {
           tp = tp+1;//making sure that there is atleast one true positive 
        }

       double prec = (double)tp/((double)(tp+fp));//System.out.println("prec:"+prec);
       double recl = (double)tp/((double)(tp+fn));//System.out.println("recl:"+recl);

       F1 = (double)(2*prec*recl)/((double)(prec+recl));

       //System.out.println("F1    :"+F1);
       System.out.println("Cur_ep:"+cr_ep);
       

       if(F1 > bstF1)
       { 
         bstF1 = F1;
         bst_ep = cr_ep;
       }
      
     }
     
     System.out.println("Best F1 Score :"+bstF1);  
      
     return bst_ep;
   }

   /*computing true positive*/
   int count_tp(int predc[],int yval[])
   {
     int count=0;

     for(int i=0;i<crow;i++)
     {
       if((predc[i]==1)&&(yval[i]==1))
        count++;       
     }

     return count;
   }

   /*computing false positive*/
   int count_fp(int predc[],int yval[])
   {
     int count=0;

     for(int i=0;i<crow;i++)
     {
       if((predc[i]==1)&&(yval[i]==0))
        count++;       
     }

     return count;
   }

   /*computing false negetive*/
   int count_fn(int predc[],int yval[])
   {
     int count=0;

     for(int i=0;i<crow;i++)
     {
       if((predc[i]==0)&&(yval[i]==1))
        count++;       
     }

     return count;
   }



   int[] cv_prediction(double cvpval[], double ep)
   {
     int val[]=new int[crow];

     for(int i=0;i<crow;i++)
     {
       if(cvpval[i]<ep)
          val[i] = 1;
       else
          val[i] = 0;
     }

     return val;
   }

   double max(double pmx[])
   {
     double temp=0.0,max=0.0;

     max = pmx[0];
     for(int i=1;i<crow;i++)
     {
       if(pmx[i]>max)
         max = pmx[i];        
     }
     return max;
   }


   double min(double pmn[])
   {
     double temp=0.0,min=0.0;

     min = pmn[0];
     for(int i=1;i<crow;i++)
     {
       if(pmn[i]<min)
         min = pmn[i];        
     }
     return min;
   }


   public static void main(String[] args)
   {
     //System.out.println("Multivariate Gaussian Calculation");
              
     AnomalyDetection ad = new AnomalyDetection();
     ad.start();
   }
}
