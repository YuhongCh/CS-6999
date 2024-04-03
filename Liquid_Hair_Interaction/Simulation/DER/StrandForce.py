import taichi as ti
import taichi.math as tm

from Liquid_Hair_Interaction.Simulation.Forces.BaseForce import BaseForce
import Liquid_Hair_Interaction.Simulation.DER.Utils as Utils

from Liquid_Hair_Interaction.Simulation.DER.Components.BendingProducts import *
from Liquid_Hair_Interaction.Simulation.DER.Components.DOFs import *
from Liquid_Hair_Interaction.Simulation.DER.Components.ElasticParameters import *
from Liquid_Hair_Interaction.Simulation.DER.Components.Kappas import *
from Liquid_Hair_Interaction.Simulation.DER.Components.MaterialFrames import *
from Liquid_Hair_Interaction.Simulation.DER.Components.ReferenceFrames import *
from Liquid_Hair_Interaction.Simulation.DER.Components.Twists import *
from Liquid_Hair_Interaction.Simulation.DER.StrandState import StrandState, StartState


@ti.data_oriented
class StrandForce(BaseForce):
    def __init__(self):
        

    std::vector<int> m_verts;  // in order root to tip
  int m_globalIndex;         // Global index amongst the hairs
  StrandParameters* m_strandParams;
  TwoDScene<3>* m_scene;
  bool m_requiresExactForceJacobian;

  // increase memory, reduce re-computation
  scalar m_strandEnergyUpdate;
  VecX m_strandForceUpdate;
  TripletXs m_strandHessianUpdate;
  SparseRXs m_hess;

  // Linear Compliant Implicit Euler solve
  VectorXs m_lambda;
  VectorXs m_lambda_v;

  //// Strand State (implicitly the end of timestep state, evolved from rest
  ///config) ////////////////////////
  StrandState* m_strandState;  // future state
  StartState* m_startState;    // current state

  //// Rest shape //////////////////////////////////////////////////////
  std::vector<scalar>
      m_restLengths;  // The following four members depend on m_restLengths,
                      // which is why updateEverythingThatDependsOnRestLengths()
                      // must be called
  scalar m_totalRestLength;
  std::vector<scalar> m_VoronoiLengths;     // rest length around each vertex
  std::vector<scalar> m_invVoronoiLengths;  // their inverses
  std::vector<scalar> m_vertexMasses;
  Vec2Array m_restKappas;
  std::vector<scalar> m_restTwists;