/* Steshenko Alexander*/

/*
This is my implementation Jacobi iterative method 
for determining the solutions of a diagonally dominant system of linear equations 
used parallel technology MPI
https://en.wikipedia.org/wiki/Jacobi_method
*/

// input: dimension, max iterations (number threads set in mpiexec or mpirun)
// output: count iteration, work time and correctness of decision (check for adequacy set max iterations)

#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
//#include <conio.h>
#include <mpi.h>

const double EPS = 0.00001;

double a[100000000];
double b[10000];
double x[10000];
double x_new[10000];

void init(int &n) {
	// uncomment for use
	/*for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j)
			scanf("%f", &a[i*n + j]);


		scanf("%f", &b[i]);

		x[i] = b[i] / a[i*n + i];
	}*/

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j) a[i*n + j] = 2.1 * (n - 1);
			else a[i*n + j] = 1.0;
		}

		if (i == n - 1) b[i] = 2.1 * (n - 1);
		else b[i] = 1.0;

		x[i] = b[i] / a[i*n + i];
		//x_new[i] = 0.0;
	}
}

bool check(int &n) {
	bool verdict = true;

	double sum = 0;
	for (int i = 0; i < n; ++i) {
		sum = 0;
		for (int j = 0; j < n; ++j) sum += a[i*n + j] * x[j];
		//std::cout << sum << " " << b[i] << std::endl;
		if (abs(sum - b[i]) >= EPS) {
			verdict = false;
			//std::cout << i << ". " << sum << " " << b[i] << " " << abs(sum - b[i]) << std::endl;
		}
	}

	return verdict;
}

int main(int argc, char* argv[])
{
	int n = atoi(argv[1]), 
		my_rank,
		n_threads,
		i, j, 
		first, 
		last, 
		comm,
		max_iter = atoi(argv[2]),
		k = 0;

	double sum, 
		x_err,
		max_x_err, 
		start_time, 
		finish_time, 
		row_err, 
		max_row_err;

	int *x_count = (int *)malloc(sizeof(int) * n),
		*x_number = (int *)malloc(sizeof(int) * n),
		*x_count2 = (int *)malloc(sizeof(int) * n),
		*x_number2 = (int *)malloc(sizeof(int) * n);

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &n_threads);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	comm = MPI_COMM_WORLD;

	if (my_rank == 0) {
		start_time = MPI_Wtime();
		init(n);
	}

	first = my_rank*n / n_threads;
	last = (my_rank + 1)*n / n_threads - 1;

	for (i = 0; i < n_threads; ++i) {
		x_count[i] = (i + 1) * n / n_threads - i*n / n_threads;
		x_number[i] = i*n / n_threads;
	}

	for (i = 0; i < n_threads; ++i) {
		x_count2[i] = x_count[my_rank];
		x_number2[i] = x_number[my_rank];
	}

	MPI_Bcast(a, n*n, MPI_DOUBLE_PRECISION, 0, comm);
	MPI_Bcast(b, n,	  MPI_DOUBLE_PRECISION, 0, comm);

	max_x_err = 0.0;
	for (i = 0; i < n; ++i) {
		x[i] = b[i] / a[i*n + i];

		if (fabs(x[i]) > max_x_err) max_x_err = fabs(x[i]);
	}

	k = 1;
	max_row_err = 0.0;

	do {
		x_err = 0.0;

		
		for (i = first; i <= last; ++i) {
			x_new[i] = b[i];
			for (j = 0; j < n; ++j) if (i != j) x_new[i] -= a[i*n + j] * x[j];
			x_new[i] /= a[i*n + i];
			
			if (x_err < abs(x_new[i] - x[i])) x_err = abs(x_new[i] - x[i]);
		}

		MPI_Allreduce(&x_err,&max_x_err, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm);

		MPI_Alltoallv(x_new, x_count2, x_number2, MPI_DOUBLE_PRECISION, x, x_count, x_number, MPI_DOUBLE_PRECISION, comm);

		row_err = 0.0;
		sum = 0;

		for (i = first; i <= last; ++i) {
			sum = b[i];
			for (j = 0; j < n; ++j) sum -= a[i*n + j] * x[j];

			if (row_err < abs(sum)) row_err = abs(sum);
		}

		MPI_Allreduce(&row_err, &max_row_err, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm);

		++k;
	} while ((max_x_err > EPS || max_row_err > EPS) && k <= max_iter);

	if (my_rank == 0) {
		finish_time = MPI_Wtime();
		printf("Count iterations: %d\n", k);
		if (check(n) == 1)
			printf("Ax==b\n");
		else
			printf("Ax!=b\n");
		printf("TimeMPI = %f\n", finish_time - start_time);
	}

	MPI_Finalize();

	//getch();
	return 0;
}

