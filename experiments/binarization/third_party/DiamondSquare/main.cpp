#include <iostream>

#include <iomanip> // used to set float precision
#include <math.h> // pow

#include <vector> //storing RectangleShape objects

#include <SFML/Graphics.hpp>
#include <string.h>
#include "DiamondSquare.h"

void displayTable(double ** map, int size)
{
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			//std::cout << map[i][j] << " " ;
			std::cout << std::setprecision(2) << std::fixed << map[i][j] << " " ;
		}
		std::cout << "" << std::endl;
	}
}

double ** gen_table(int factor)
{
	int size = factor;

	double ** map = new double *[size];
	for(int i = 0; i < size; i++)
	{
		map[i] = new double[size];
		for(int j = 0; j < size; j++)
			map[i][j] = 0.0;
	}
	return map;
}

int main(int argc, char ** argv)
{
	double range = 5; // +-10

	int n = 2;
	if(argc>=2){
		n = atoi(argv[1]);
	}
	int size = pow(2, n)+1;

	//std::cout << "Size : " << size << std::endl;
	
	double ** map1 = gen_table(size);
	
	DiamondSquare ds(map1, size);
	double ** map = ds.process();
        if(argc == 3 && strcmp(argv[2],"no_io")==0){
		return 0;
	}	

	/*
	 * Code that follows is a DiamondSquare algorithm test case using SFML
	 * to display a generated heightmap.
	 */

	//display it :
	int size_bloc = 2;
	sf::RenderWindow window(sf::VideoMode(size_bloc*size, size_bloc*size), "Diamond Square");
	
	//storing shapes in vector (the very bad way)
	std::vector<std::vector<sf::RectangleShape> > shapes;
	for(int i = 0; i < size; i++)
	{
		std::vector<sf::RectangleShape> sub_shapes;
		for(int j = 0; j < size; j++)
		{
			sf::RectangleShape rs(sf::Vector2f(size_bloc, size_bloc));
			rs.setFillColor(sf::Color(
					(int)map[i][j], 
					(int)map[i][j], 
					(int)map[i][j]));
				rs.setPosition(i*size_bloc, j*size_bloc);
			sub_shapes.push_back(rs);
		}
		shapes.push_back(sub_shapes);
	}
	
	while(window.isOpen())
	{
		sf::Event event;
		while(window.pollEvent(event))
		{
			if(event.type == sf::Event::Closed)
				window.close();
		}
		window.clear();
		for(int i = 0; i < size; i++)
		{
			for(int j = 0; j < size; j++)
			{
				window.draw(shapes[i][j]);
			}
		}
		window.display();
	}
	
	return 0;
}
