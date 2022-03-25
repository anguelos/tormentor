class DiamondSquare
{
	private:
		double random_range;
		double min_val;
		double max_val;

		double ** map;
		int size;

		int range;

	public:
		DiamondSquare(double ** array, int s);
		~DiamondSquare();

		double ** process();
		void _on_start();
		void _on_end();
		void diamondStep(int, int);
		void squareStep(int, int);

		double dRand(double dMin, double dMax);
};
