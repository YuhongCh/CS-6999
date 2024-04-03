import taichi as ti


@ti.data_oriented
class BaseForce:
    def __init__(self):
        self.pos_index = 0
        self.vel_index = 0
        self.J_index = 0
        self.Jv_index = 0
        self.Jxv_index = 0
        self.tildeK_index = 0

    @ti.func
    def t_addEnergy()

 public:
  virtual ~Force();

  virtual void addEnergyToTotal(const VectorXs& x, const VectorXs& v,
                                const VectorXs& m, scalar& E) {}
  virtual void addGradEToTotal(const VectorXs& x, const VectorXs& v,
                               const VectorXs& m, VectorXs& gradE) {}

  virtual void addHessXToTotal(const VectorXs& x, const VectorXs& v,
                               const VectorXs& m, TripletXs& hessE) {}

  virtual void addHessVToTotal(const VectorXs& x, const VectorXs& v,
                               const VectorXs& m, TripletXs& hessE) {}

  virtual void addGradEToTotal(const VectorXs& x, const VectorXs& v,
                               const VectorXs& m, VectorXs& gradE, int pidx) {}

  virtual void addHessXToTotal(const VectorXs& x, const VectorXs& v,
                               const VectorXs& m, VectorXs& hessE, int pidx) {}

  virtual void addHessVToTotal(const VectorXs& x, const VectorXs& v,
                               const VectorXs& m, VectorXs& hessE, int pidx) {}

  virtual void computeIntegrationVars(const VectorXs& x, const VectorXs& v,
                                      const VectorXs& m, VectorXs& lambda,
                                      VectorXs& lambda_v, TripletXs& J,
                                      TripletXs& Jv, TripletXs& Jxv,
                                      TripletXs& tildeK, TripletXs& stiffness,
                                      TripletXs& damping, VectorXs& Phi,
                                      VectorXs& Phiv, const scalar& dt) {}

  virtual int numConstraintPos() = 0;

  virtual int numConstraintVel() = 0;

  virtual int numJ() = 0;

  virtual int numJv() = 0;

  virtual int numJxv() = 0;

  virtual int numTildeK() = 0;

  virtual bool isParallelized() = 0;

  virtual bool isPrecomputationParallelized() = 0;

  virtual void storeLambda(const VectorXs& lambda, const VectorXs& lambda_v);

  virtual void setInternalIndex(int index_pos, int index_vel, int index_J,
                                int index_Jv, int index_Jxv, int index_tildeK);

  virtual Force* createNewCopy() = 0;

  virtual const char* name() = 0;

  virtual void getAffectedVars(int pidx, std::unordered_set<int>& vars) = 0;

  virtual int getAffectedHair(const std::vector<int> particle_to_hairs);

  virtual bool isContained(int pidx) = 0;

  virtual bool isExternal();

  virtual void preCompute(const VectorXs& x, const VectorXs& v,
                          const VectorXs& m, const scalar& dt);

  virtual bool isInterHair() const;

  virtual void postStepScene(const scalar& dt);
};