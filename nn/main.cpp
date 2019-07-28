//#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <cmath>
//#include <math.h>




//using namespace Eigen;
struct Connection
{
	double weight;
	double deltaweight;
};


//Neuron CLASS**********************************

class Neuron
{
public:
	Neuron(unsigned numoutputs, unsigned myindex);
	void feedforward(const Layer& prevLayer);
	void setOutputval(double val) { m_outputval = val;}
	double getOutputval(void) const { return m_outputval; }
	void calcoutputgradients(double targetvals);


private:
	double m_outputval;
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);

	static double randomweight(void) { return rand() / double(RAND_MAX); }
	std::vector<Connection> m_outputweight;
	unsigned m_myindex;
	double m_gradient;
};


void Neuron::calcoutputgradients(double targetvals)
{
	double delta = targetval - m_outputval;
	m_gradient = delta * transferFunctionDerivative(m_outputval);
}


double Neuron::transferFunction(double x)
{
	//tahn
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	return 1.0 - x * x; 
}


void Neuron::feedforward(const Layer& prevLayer)
{
	/*double sum = 0.0;

	for (unsigned  n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputval() * prevLayer[n].m_outputweight[m_myindex].weight;
	}
	m_outputval = Neuron::transferFunction(sum);
	*/
}

Neuron::Neuron(unsigned numoutputs, unsigned myindex)
{
	for (unsigned c = 0; c < numoutputs; ++c)
	{
		m_outputweight.push_back(Connection());
		m_outputweight.back().weight = randomweight();
	}
	m_myindex = myindex;
}
typedef std::vector<Neuron> Layer;











//NET CLASS**********************************


class net 
{
public:
	net(std::vector<unsigned>&topology);
	void feedforward(const std::vector<double>& inputvals) {};
	void backprop(const std::vector<double>& targetvals) {};
	void getresults(std::vector<double>& resultsvals) {};

private:
	//std::vector<layer> m_layer;
	std::vector<Layer> m_layer;
	double m_error;
	double m_recentavgerror;
	double m_recentavgsmoothfactor;
};


void net::backprop(const std::vector<double>& targetvals) 
{
	//cal overall erorr
	
	// for all layers in  from outputs to first hidden layer 
	//update connection  

	Layer& outputlayer = m_layers.back();
	m_error = 0.0;
	for (unsigned n = 0; n < outputlayer.size() - 1; ++n)
	{
		double delta = targetvals[n] - outputlayer[n].getOutputval();
		m_error += delta * delta;

	}
	m_error /= outputlayer.size() - 1;
	m_error = sqrt(m_error);
	// cal output gradinets
	for (unsigned n = 0; n < outputlayer.size() - 1; ++n)
	{
		outputlayer[n].calcoutputgradients(targetvals[n]);
	}

	// cal grandinets on hidden layer
	for (unsigned layernum = m_layer.size() - 2; layernum > 0; --layernum)
	{
		Layer& hiddenLayer = m_layer[layernum];
		Layer& nextlayer = m_layer[layernum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calchiddengradients(nextlayer);
		}

	}
	for (unsigned layernum = m_layer.size() - 1; layernum > 0; --layernum)
	{
		Layer& layer = m_layer[layernum];
		Layer& prevlayer = m_layer[layernum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateinputweight(prevlayer);
		}
	}
	m_recentavgerror = (m_recentavgerror * m_recentavgsmoothfactor + m_error) / (m_recentavgsmoothfactor + 1.0);

}



void net::feedforward(const std::vector<double>& inputvals)
{
	assert(inputvals.size() == m_layer[0].size() - 1);
	// this puts the inputs into the neuron
	for (unsigned i = 0; i < inputvals.size(); ++i)
	{
		m_layer[0][i].setOutputval(inputvals[i]);
	}
	//forward propagate
	for (unsigned layernum = 1; layernum < m_layer.size(); ++layernum)
	{
		Layer& prevLayer = m_layer[layernum - 1];
		for (unsigned n = 0; n < m_layer[layernum].size() - 1;  ++n)
		{
			m_layer[layernum][n].feedforward(prevLayer);
		}
	}
}


net::net( std::vector<unsigned>& topology)
	{
		unsigned numlayers = topology.size();
		for (unsigned layernum = 0; layernum < numlayers; ++layernum)
		{
		m_layer.push_back(Layer());
		unsigned numoutputs = layernum == topology.size() - 1 ? 0 : topology[layernum + 1];

			for (unsigned neuronnum = 0; neuronnum <= topology[layernum]; ++neuronnum)
			{
				m_layer.back().push_back(Neuron(numoutputs, neuronnum));
				std::cout << "made a neuron" << std::endl;	
			}
		}
	}

/*void net::backprop(const std::vector<double>& targetvals)
{
	//calc the overall net error ()rms of outputs  neurons errors 
	
}*/
/*double random(double x)
{
	return (double)(rand() % 10000 + 1) / 10000 - 0.5;
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));

	std::cout << x << std::endl;
}*/


int main()
{
	std::vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(2);

	net mynet (topology);

	std::vector<double> inputvals;
	mynet.feedforward(inputvals);

	std::vector<double> targetvals;
	mynet.backprop(targetvals);

	std::vector<double> resultsvals;
	mynet.getresults(resultsvals);


	/*mynet.feedforward(inputvals);
	mynet.backprop(targetvals);
	mynet.getresults(resultsvals);*/


	


	
	


	//dynamic matrix
	/*Matrix3d training_inputs;
	training_inputs <<
		0, 1, 1,
		1, 1, 1,
		0, 1, 0;*/

	//std::cout << training_inputs.proforward(0.7) << std::endl;


	/*MatrixXd training_outputs = MatrixXd::Random(1, 4);
	training_outputs << 0, 1, 1, 0;*/

	//srand(time(0));  // Initialize random number generator.
	//std::cout << training_inputs << std::endl;
	//std::cout << training_outputs << std::endl;
	
	puts("hello");
}
