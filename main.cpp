#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Network/IOService.h>
#include <cstring>
#include <libOTe/Tools/LDPC/Util.h>
#include <libOTe_Tests/Common.h>
#include <libdivide.h>
#include <map>
#include <vector>
#include "Defines.h"
#include "Paxos.h"
#include "PaxosImpl.h"
using namespace oc;
using namespace volePSI;
using namespace osuCrypto;
using namespace std;

void perfBaxos(oc::CLP &cmd)
{
    auto n = cmd.getOr("n", 1ull << cmd.getOr("nn", 26));
    auto t = cmd.getOr("t", 1ull);
    // auto rand = cmd.isSet("rand");
    auto v = cmd.getOr("v", cmd.isSet("v") ? 1 : 0);
    auto w = cmd.getOr("w", 3);
    auto ssp = cmd.getOr("ssp", 40);
    auto dt = cmd.isSet("binary") ? PaxosParam::Binary : PaxosParam::GF128;
    auto nt = cmd.getOr("nt", 1);

    // PaxosParam pp(n, w, ssp, dt);
    auto binSize = 1 << cmd.getOr("lbs", 14);
    u64 baxosSize;
    {
        Baxos paxos;
        paxos.init(n, binSize, w, ssp, dt, oc::ZeroBlock);
        baxosSize = paxos.size();
    }
    std::vector<block> key(n), val(n), pax(baxosSize);

    auto decode_size = 500;

    std::vector<block> decode_val(decode_size);
    std::vector<block> decode_key(decode_size);

    PRNG prng(ZeroBlock);
    prng.get<block>(key);
    prng.get<block>(val);

    Timer timer;
    auto start = timer.setTimePoint("start");
    auto end = start;
    auto decode_start = start;
    auto decode_end = start;
    for (u64 i = 0; i < t; ++i) {
        Baxos paxos;

        paxos.init(n, binSize, w, ssp, dt, block(i, i));

        // if (v > 1)
        //	paxos.setTimer(timer);

        timer.setTimePoint("s" + std::to_string(i));

        paxos.solve<block>(key, val, pax, nullptr, nt);

        end = timer.setTimePoint("d" + std::to_string(i));

        memcpy(decode_key.data(), key.data(), decode_key.size() * sizeof(block));

        // paxos.decode<block>(decode_key, decode_val, pax, nt);

        vector<block> hashs(decode_key.size());
        vector<vector<u64>> idxs(decode_key.size(), vector<u64>(3 + 1));

        paxos.computeRetrivalIdx<block>(decode_key, hashs, idxs, nt);

        // for (auto &vec : idxs) {
        //     for (auto &u : vec)
        //         std::cout << u << ",";
        // }

        decode_start = timer.setTimePoint("decode");

        vector<std::map<u64, block>> pp(paxos.mNumBins, std::map<u64, block>());

        // dense part
        for (u64 b = 0; b < paxos.mNumBins; ++b) {
            auto paxos_seg = std::vector<block>(
                pax.begin() + b * (paxos.mPaxosParam.mSparseSize + paxos.mPaxosParam.mDenseSize),
                pax.begin() + (b + 1) * (paxos.mPaxosParam.mSparseSize + paxos.mPaxosParam.mDenseSize));

            for (u64 j = 0; j < paxos.mPaxosParam.mDenseSize; ++j) {
                pp[b][j + paxos.mPaxosParam.mSparseSize] = paxos_seg[j + paxos.mPaxosParam.mSparseSize];
            }
        }

        // sparse part
        for (u64 i = 0; i < decode_key.size(); ++i) {
            auto vec = idxs[i];
            auto binIdx = vec[paxos.mPaxosParam.mWeight];
            for (u64 j = 0; j < paxos.mPaxosParam.mWeight; ++j) {
                pp[binIdx][vec[j]] = pax[binIdx * (paxos.mPaxosParam.mSparseSize + paxos.mPaxosParam.mDenseSize) + vec[j]];
            }
        }

        decode_end = timer.setTimePoint("decode");

        std::cout << "total dense size: " << paxos.mNumBins * paxos.mPaxosParam.mDenseSize * sizeof(block) / 1024.0 / 1024.0 << " MB" << std::endl;

        paxos.decodeRetrivalIdx<block>(hashs, idxs, decode_val, pp, nt);
    }

    if (memcmp(val.data(), decode_val.data(), decode_val.size() * sizeof(block)) != 0)
        std::cout << "Error: Decoded values do not match original values." << std::endl;

    if (v)
        std::cout << timer << std::endl;

    auto tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / double(1000);

    auto decode_t = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count() / double(1000);

    std::cout << "encode " << tt << " ms, decode " << decode_t << " ms, e=" << double(baxosSize) / n << std::endl;
}

void testGen(oc::CLP &cmd)
{
    auto n = cmd.getOr("n", 1ull << cmd.getOr("nn", 10));
    auto t = cmd.getOr("t", 1ull);
    std::vector<block> key(n);
    PRNG prng(ZeroBlock);
    Timer timer;
    auto start = timer.setTimePoint("start");
    auto end = start;
    for (u64 i = 0; i < t; ++i) {
        prng.get<block>(key);
        end = timer.setTimePoint("d" + std::to_string(i));
    }

    // std::cout << timer << std::endl;
    auto tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / double(1000);
    std::cout << "total " << tt << "ms" << std::endl;
}

void testAdd(oc::CLP &cmd)
{
    auto n = cmd.getOr("n", 1ull << cmd.getOr("nn", 10));
    auto t = cmd.getOr("t", 1ull);
    std::vector<block> key(n), key1(n);
    std::cout << "Size of int: " << sizeof(block) << " bytes" << std::endl;
    PRNG prng(ZeroBlock);
    prng.get<block>(key);
    prng.get<block>(key1);
    Timer timer;
    auto start = timer.setTimePoint("start");
    auto end = start;
    for (u64 i = 0; i < t; ++i) {
        for (u64 j = 0; i < n; ++i) {
            key[j] = key[j] + key1[j];
        }
        end = timer.setTimePoint("d" + std::to_string(i));
    }

    // std::cout << timer << std::endl;
    auto tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / double(1000);
    std::cout << "total " << tt << "ms" << std::endl;
}

template <typename T>
void perfPaxosImpl(oc::CLP &cmd)
{
    auto n = cmd.getOr("n", 1ull << cmd.getOr("nn", 10));
    u64 maxN = std::numeric_limits<T>::max() - 1;
    auto t = cmd.getOr("t", 1ull);
    // auto rand = cmd.isSet("rand");
    auto v = cmd.getOr("v", cmd.isSet("v") ? 1 : 0);
    auto w = cmd.getOr("w", 3);
    auto ssp = cmd.getOr("ssp", 40);
    auto dt = cmd.isSet("binary") ? PaxosParam::Binary : PaxosParam::GF128;
    auto cols = cmd.getOr("cols", 0);

    PaxosParam pp(n, w, ssp, dt);
    // std::cout << "e=" << pp.size() / double(n) << std::endl;
    if (maxN < pp.size()) {
        std::cout << "n must be smaller than the index type max value. " LOCATION << std::endl;
        throw RTE_LOC;
    }

    auto m = cols ? cols : 1;
    std::vector<block> key(n);
    oc::Matrix<block> val(n, m), pax(pp.size(), m);
    PRNG prng(ZeroBlock);
    prng.get<block>(key);
    prng.get<block>(val);

    Timer timer;
    auto start = timer.setTimePoint("start");
    auto end = start;
    for (u64 i = 0; i < t; ++i) {
        Paxos<T> paxos;
        paxos.init(n, pp, block(i, i));

        if (v > 1)
            paxos.setTimer(timer);

        if (cols) {
            paxos.setInput(key);
            paxos.template encode<block>(val, pax);
            timer.setTimePoint("s" + std::to_string(i));
            paxos.template decode<block>(key, val, pax);
        } else {
            paxos.template solve<block>(key, oc::span<block>(val), oc::span<block>(pax));
            timer.setTimePoint("s" + std::to_string(i));
            paxos.template decode<block>(key, oc::span<block>(val), oc::span<block>(pax));
        }

        end = timer.setTimePoint("d" + std::to_string(i));
    }

    if (v)
        std::cout << timer << std::endl;

    auto tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / double(1000);
    std::cout << "total " << tt << "ms" << std::endl;
}

int main(int argc, char **argv)
{
    CLP cmd;
    cmd.parse(argc, argv);
    perfBaxos(cmd);
    // testGen(cmd);
    // testAdd(cmd);
    return 0;
}