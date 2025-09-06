#pragma once

#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Network/IOService.h>
#include <cstring>
#include <map>
#include <vector>
#include "Defines.h"
#include "Paxos.h"
using namespace oc;
using namespace volePSI;
using namespace osuCrypto;
using namespace std;

// A sparse OKVS based on Paxos, including encoding, decoding, and computing sparse index.
class SparseOKVS {
public:
    u64 sparseSize;
    u64 denseSize;
    u64 weight;
    u64 binNum;

    SparseOKVS(u64 numItems, u64 weight_ = 3, u64 ssp = 40, u64 binSize_ = 1 << 14);

    void encode(vector<block> keys, vector<block> values, vector<block> E, u64 numThreads);

    vector<block> getDense(vector<block> &E);

    void computeIndex(vector<block> keys, vector<block> &hashs, vector<vector<u64>> &idxs, u64 numThreads);

    void decode(vector<block> &hashs, vector<vector<u64>> &idxs, vector<block> &values, vector<std::map<u64, block>> E, u64 numThreads);

    u64 size();

    u64 sizeOfSparse();

    u64 sizeOfDense();

private:
    Baxos paxos;
    PaxosParam param;
};
