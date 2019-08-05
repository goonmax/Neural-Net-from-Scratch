//#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <sstream>
//#include <math.h>



class Trainingdata
{
public: 
	Trainingdata(const std::string filename);
	bool isEof(void) { return m_trainingDataFile.eof();  }
	void geTopology(std::vector<unsigned>& topology);
	unsigned getNextInputs(std::vector<double>& inputVals);
	unsigned getTargetout(std::vector<double>& targetOutputVals);

private:
	std::ifstream m_trainingDataFile;
};

void Trainingdata::geTopology(std::vector<unsigned>& topology)
{
	std::string line;
	std::string label;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology: ") != 0) {
		abort();
	}

	while (!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);

	}
	return;
}


Trainingdata::Trainingdata(const std::string filename)
{
	m_trainingDataFile.open(filename.c_str());
}



unsigned Trainingdata::getNextInputs(std::vector<double>& inputsVals)
{
	inputsVals.clear();
	std::string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);
	std::string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double onevalue;
		while (ss >> onevalue) {
			inputsVals.push_back(onevalue);

		}
	}
	return inputsVals.size();

}
unsigned Trainingdata::getTargetout(std::vector<double>& targetOutputVals)
{
	targetOutputVals.clear();
	std::string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double onevalue;
		while (ss >> onevalue) {
			targetOutputVals.push_back(onevalue);
		}

	}
	return targetOutputVals.size();
}
//using namespace Eigen;
struct Connection
{
	double weight;	
	double deltaweight;
};


//Neuron CLASS**********************************
typedef std::vector<Neuron> Layer;

class Neuron
{
public:
	Neuron(unsigned numoutputs, unsigned myindex);
	void feedforward(const Layer& prevLayer);
	void setOutputval(double val) { m_outputval = val;}
	double getOutputval(void) const { return m_outputval; }
	void calcoutputgradients(double targetvals);
	void calchiddengradients(const Layer& nextlayer);
	void updateinputweight(Layer& prevlayer);



private:
	static double eta, alpha;
	double m_outputval;
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double sumDOW(const Layer& nextlayer) const;
	static double randomweight(void) { return rand() / double(RAND_MAX); }
	std::vector<Connection> m_outputweight;
	unsigned m_myindex;
	double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateinputweight(Layer& prevlayer)
{
	//weights are stored here and upodated 
	for (unsigned n = 0; n < prevlayer.size(); ++n)
	{
		Neuron& neuron = prevlayer[n];
		double oldDeltaWeight = neuron.m_outputweight[m_myindex].deltaweight;
		//individual	input magnified by the gradient and train rate

		double newDeltaweight = eta * neuron.getOutputval() * m_gradient + alpha * oldDeltaWeight;
		neuron.m_outputweight[m_myindex].deltaweight = newDeltaweight;
		neuron.m_outputweight[m_myindex].weight += newDeltaweight;
	}


}





double Neuron::sumDOW(const Layer& nextlayer) const
{
	double sum = 0.0;
	for (unsigned n = 0; n < nextlayer.size() - 1; ++n)
	{
		sum += m_outputweight[n].weight * nextlayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calchiddengradients(const Layer& nextlayer)
{
	double dow = sumDOW(nextlayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputval);

}


void Neuron::calcoutputgradients(double targetvals)
{
	double delta = targetvals - m_outputval;
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











//NET CLASS**********************************


class net 
{
public:
	net(std::vector<unsigned>&topology);
	void feedforward(const std::vector<double>& inputvals);
	void backprop(const std::vector<double>& targetvals);
	void getresults(std::vector<double>& resultsvals) const;
	double getRecentAverageError(void) const { return m_recentavgerror; }

private:
	//std::vector<layer> m_layer;
	std::vector<Layer> m_layer;
	double m_error;
	double m_recentavgerror;
	double m_recentavgsmoothfactor;
};
void net::getresults(std::vector<double>& resultsvals) const
{
	resultsvals.clear();

	for (unsigned n = 0; n < m_layer.back().size() - 1; ++n)
	{
		resultsvals.push_back(m_layer.back()[n].getOutputval());
	}
}

void net::backprop(const std::vector<double>& targetvals) 
{
	//cal overall erorr
	
	// for all layers in  from outputs to first hidden layer 
	//update connection  

	Layer& outputlayer = m_layer.back();
	m_error = 0.0;
	for (unsigned n = 0; n < outputlayer.size() - 1; ++n)
	{
		double delta = targetvals[n] - outputlayer[n].getOutputval();
		m_error += delta * delta;

	}
	m_error /= outputlayer.size() - 1;
	m_error = sqrt(m_error);
	m_recentavgerror = (m_recentavgerror * m_recentavgsmoothfactor + m_error) / (m_recentavgsmoothfactor + 1.0);
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
	// for all layers in  from outputs to first hidden layer 
	

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
			m_layer.back().back().setOutputval(1.0);
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


void showVectorVals(std::string label, std::vector<double>& v)
{
	std::cout << label << "";
	for (unsigned i = 0; i < v.size(); ++i)
	{
		std::cout << v[i] << "";
	}
	std::cout << std::endl;
}

int main()
{
	Trainingdata trainData("");
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

	std::vector<unsigned> topology;
	trainData.geTopology(topology);
	net mynet(topology);
	std::vector<double> inputvals, targetvals, resultsvals;
	int trainingPass = 0;
	while (!trainData.isEof()) {
		++trainingPass;
		std::cout  << "Pass" << trainingPass;
		//get new data from and feed it forward 

		if (trainData.getNextInputs(inputvals) != topology[0]) { break; }
		showVectorVals(": Inputs:", inputvals);
		mynet.feedforward(inputvals);

		mynet.getresults(resultsvals);
		showVectorVals(": Outputs:", resultsvals);

		trainData.getTargetout(targetvals);
		showVectorVals("Targets", targetvals);
		assert(targetvals.size() == topology.back());

		mynet.backprop(targetvals);

		std::cout << "Net recent average error:" << mynet.getRecentAverageError() << std::endl;

	}


	std::cout << "Man I wish I was good at this" << std::endl;
	


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
