#ifndef PQUBIT_H
#define PQUBIT_H

#include <array>
#include <tuple>
#include <vector>
#include <ostream>
#include <map>
#include <set>
#include <unordered_map>

namespace pgq{
    int pegasus_0_shift[] = {2, 2, 10, 10, 6, 6, 6, 6, 2, 2, 10, 10};

    std::tuple<int, int, int> vert2horz(int w, int k, int z){
        int t = k / 4;
        int xv = 3 * w + t;
        int yv = 2 + 3*z + (2*t) % 3;

        int z2 = (xv - 1) / 3;
        int w2 = yv / 3;
        int k02 = ( yv % 3 ) * 4;

        return std::make_tuple(w2, k02, z2);
    }
    std::tuple<int, int, int> horz2vert(int w, int k, int z){
        int t = k / 4;
        int xh = 1 + 3*z + (2 * (t+2)) % 3;
        int yh = 3*w + t;

        int z2 = (yh -2) / 3;
        int w2 = xh / 3;
        int k02 = (xh % 3) * 4;

        return std::make_tuple(w2, k02, z2);
    }

    std::tuple<int, int, int> internal_coupling(int u, int w, int k, int z, int j){
        if (u == 0){
            int d1 = ( j < pegasus_0_shift[k/2] ? 1 : 0);
            int d2 = ( k < pegasus_0_shift[6 + j/2] ? 1 : 0);
            return std::make_tuple(z+d1, j, w-d2);
        } else {
            int d1 = (k < pegasus_0_shift[j/2] ? 1 : 0);
            int d2 = (j < pegasus_0_shift[6 + k/2] ? 1 : 0);
            return std::make_tuple(z+d2, j, w-d1);
        }
    }

    // Thin wrappen around an array of 8 ints specifying the indices of a cell
    struct qcell{
        int idxs[8];
        int& operator[](std::size_t i){return idxs[i];}
        const int& operator[](std::size_t i) const {return idxs[i];}
    };

    struct Pcoord{
        int t, x, z;
        Pcoord(int t, int x, int z): t(t), x(x), z(z) {}
        inline bool operator==(const Pcoord& p) const { 
            return t == p.t && x == p.x && z == p.z;
        }
    };

    class Pqubit{
    public:
        int m;
        int u, w, k, z;

        Pqubit() = default;
        Pqubit(int m, int u, int w, int k, int z): m(m), u(u), w(w), k(k), z(z) { }
        bool operator==(const Pqubit &q){
            return m == q.m && u == q.u && w == q.w && k == q.k && z == q.z;
        }
        bool operator!=(const Pqubit &q){
            return !(*this == q);
        }
        friend std::ostream& operator<<(std::ostream& os, const Pqubit& q);

        int to_linear() const{
            return z + (m-1)*(k + 12 * (w + m + u));
        }
        Pqubit conn_external(int dz=1) const{
            return {m, u, w, k, z + dz};
        }
        Pqubit conn_odd() const{
            int k2 = (k % 2 == 0 ? k + 1 : k - 1);
            return {m, u, w, k2, z};
        }
        bool is_vert_coord() const{
            return u == 0;
        }
        bool is_horz_coord() const{
            return u == 1;
        }
        Pqubit conn_k44(int dk) const{
            int w2, k02, z2;
            std::tie(w2, k02, z2) = (is_vert_coord() ? vert2horz(w, k, z) : horz2vert(w, k, z));
            return {m, 1-u, w2, k02+dk, z2};
        }
        Pqubit conn_internal(int dk) const{
            int k0_cluster;
            std::tie(std::ignore, k0_cluster, std::ignore) =
                    (is_vert_coord() ? vert2horz(w, k, z) : horz2vert(w, k, z));
            int j = (12 + k0_cluster + dk) % 12;
            int w2, k2, z2;
            std::tie(w2, k2, z2) = internal_coupling(u, w, k, z, j);
            return {m, 1-u, w2, k2, z2};
        }
        // conn_internal_abs
        inline qcell k44_qubits() const{
            qcell k44_arr{};
            if( is_vert_coord()){
                Pqubit q2_arr[4];
                for(int i = 0; i < 4; ++i){
                    q2_arr[i] = conn_k44(i);
                    k44_arr[4+i] = q2_arr[i].to_linear();
                }
                for(int i = 0; i < 4; ++i){
                    k44_arr[i] = q2_arr[0].conn_k44(i).to_linear();
                }
            } else {
                Pqubit q2_arr[4];
                for(int i = 0; i < 4; ++i){
                    q2_arr[i] = conn_k44(i);
                    k44_arr[i] = q2_arr[i].to_linear();
                }
                for(int i = 0; i < 4; ++i){
                    k44_arr[4+i] = q2_arr[0].conn_k44(i).to_linear();
                }
            }
            return k44_arr;
        }
    };

    std::vector<qcell> generate_regular_cell_grid(int m){
        int w0[3] = {1, 0, 0};

        std::vector<qcell> v;
        v.reserve(3*(m-1)*(m-1));

        for(int t = 0; t < 3; ++t){
            for(int w = w0[t]; w < m - 1 + w0[t]; ++w){
                //int x = w - w0[t];
                for(int z = 0; z < m-1; ++z){
                    int k = 4*t;
                    Pqubit q0{m, 0, w, k, z};
                    qcell k44 = q0.k44_qubits();
                    v.push_back(k44);
                }
            }
        }

        return v;
    }

    std::pair<std::map<std::tuple<int, int, int>, qcell>, int > collect_available_unit_cells(
        int m, std::set<int> nodes, std::set<std::pair<int, int>> edges
    ){

    }

    std::ostream& operator<<(std::ostream& os, const Pqubit& q){
        if(q.is_vert_coord()){
            return os << "Vert(M="<<q.m<<"[u=0, w: "<<q.w<<", k: "<<q.k<<", z: "<<q.z<<"])";
        }
        else{
            return os << "Horz(M="<<q.m<<"[u=1, w: "<<q.w<<", k: "<<q.k<<", z: "<<q.z<<"])";
        }
    }
}

namespace std{
    template<> struct hash<pgq::Pcoord> {
        inline size_t operator()(const pgq::Pcoord& p) const{
            hash<int> int_hasher;
            return int_hasher(p.t) ^ int_hasher(p.x) ^ int_hasher(p.z);
        }
    };
}

#endif