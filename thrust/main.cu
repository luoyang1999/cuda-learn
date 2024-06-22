#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


int main() {
  // Generate random data serially.
  thrust::host_vector<int> H(4);
  std::vector<int> I(4);

  // init
  H[0] = 14;
  H[1] = 20;
  H[2] = 38;
  H[3] = 46;
  std::cout << "H has size " << H.size() << std::endl;

  for(int i=0;i<4;i++){
    std::cout << "H[" <<i << "]=" << H[i] << std::endl;
  }

  // resuze
  H.resize(5);
  std::cout << "H has size " << H.size() << std::endl;

  // copy host_vector h to device_vector d
  thrust::device_vector<int> D = H;
  for(int i=0;i<D.size();i++){
    std::cout << "D[" <<i << "]=" << D[i] << std::endl;
  }

  D[0] = 99;
  D[1] = 88;

  for(int i=0;i<H.size();i++){
    std::cout << "H[" <<i << "]=" << H[i] << std::endl;
  }

  // sequence
  std::cout<< "thrust::sequence" << std::endl;
  H.resize(8);
  // 从0开始一直到n-1
  thrust::sequence(H.begin(), H.end());
  for(int i=0;i<H.size();i++){
    std::cout << "H[" <<i << "]=" << H[i] << std::endl;
  }

  // sequence
  std::cout<< "thrust::fill" << std::endl;
  // 从begin到end赋值
  thrust::fill(D.begin(), D.begin()+2, 8);
  for(int i=0;i<D.size();i++){
    std::cout << "D[" <<i << "]=" << D[i] << std::endl;
  }

  // transform
  std::cout<<"transform"<<std::endl;
  thrust::device_vector<int> Y(10);
  // 将D从头到尾反向，然后赋值给Y
  thrust::transform(D.begin(), D.end(), Y.begin(), thrust::negate<int>());
  for(int i=0;i<Y.size();i++){
    std::cout << "Y[" <<i << "]=" << Y[i] << std::endl;
  }

    // 

}
