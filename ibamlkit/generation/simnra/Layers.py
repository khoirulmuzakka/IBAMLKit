#!/usr/bin/env python3

# @file Layers.py
# First written by Khoirul Faiq Muzakka on 22.12.2022
# @brief

# ElementPoly.pyx
from .Exceptions import ParameterNegativeError, SIMNRAError, ParameterNotFoundError, ThicknessOutOfRangeError, LayerNotNormalizedError
from .SIMNRA import SIMNRA  # Importing SIMNRA module
import os
import shutil
from pathlib import Path
import numpy as np
from .Settings import SettingsApp
import tempfile
from .Parameterization import *

class SIMNRALayers:
    """ 
    This class wrap the SIMNRA object and provides functionality to implement the standard and polynomial parameterization to the layers. 
    """
    def __init__(self, settings, mode="RBS"):
        """ 
        class constructor. 
        Arguments : 
            - Settings : a SettingsApp object 
            - mode : a string that indicate IBA mode
        """
        self.settings = settings
        self.checkConSum = False
        self.mode = mode
        self.createSIMNRAobject()

        self.Nlayers = self.simNRA.Target.NumberOfLayers
        self.lr = self.settings.LayerSettings.getSettings("UseLayerRoughness")
        self.lp = self.settings.LayerSettings.getSettings("UseLayerPorosity")

        if len(self.lr) != self.Nlayers:
            self.lr = [False for i in range(self.Nlayers)]

        if len(self.lp) != self.Nlayers:
            self.lp = [False for i in range(self.Nlayers)]

        self.NelPerLay = []
        self.ElPerLayer = []
        self.elZdict = {}
        for i in range(self.Nlayers):
            self.NelPerLay.append(self.simNRA.Target.NumberOfElements(i + 1))
            elnames = []
            for j in range(1, self.simNRA.Target.NumberOfElements(i + 1) + 1):
                ele = str(self.simNRA.Target.ElementName(i + 1, j)).strip() 
                self.elZdict[ele] = int(self.simNRA.Target.ElementZ(i + 1, j))
                elnames.append(ele)
            self.ElPerLayer.append(elnames)

        self.ionEnergy = self.simNRA.Setup.Energy
        self.ionMass = self.simNRA.Projectile.Mass
        self.ionZ = self.simNRA.Projectile.Charge
        self.EIonMin = self.settings.FitSettings.getSettings("EIonMin")
        self.checkRange = self.settings.FitSettings.getSettings("RangeChecker")
        self.first = True

        # Initiate all fittable parameters
        self.parametersMap = {}
        self.paramBounds = {}
        self.parametersMap["ParticlesSr_" + mode] = self.simNRA.Setup.ParticlesSr
        self.parametersMap["Calib_Offset_"+mode] = self.simNRA.Setup.CalibrationOffset
        self.parametersMap["Calib_Linear_"+mode] = self.simNRA.Setup.CalibrationLinear
        self.parametersMap["Calib_Quadratic_"+mode] = self.simNRA.Setup.CalibrationQuadratic
        self.parametersMap["FWHM_" + mode] = self.simNRA.Setup.DetectorResolution
        self.parametersMap["BeamEnergy_" + mode] = self.simNRA.Setup.Energy
        self.parametersMap["BeamSpread_" + mode] = self.simNRA.Setup.Beamspread

        self.paramBounds["ParticlesSr_" + mode] = (0.1*self.parametersMap["ParticlesSr_" + mode], 10*self.parametersMap["ParticlesSr_" + mode]) 
        self.paramBounds["Calib_Offset_"+mode] = (-20.0, 20.0)
        self.paramBounds["Calib_Linear_"+mode] = (0.95*self.parametersMap["Calib_Linear_"+mode], 1.05*self.parametersMap["Calib_Linear_"+mode] )
        self.paramBounds["Calib_Quadratic_"+mode] = (-0.01, 0.01)
        self.paramBounds["FWHM_" + mode] = (0.0, 2*self.parametersMap["FWHM_" + mode])
        self.paramBounds["BeamEnergy_" + mode] = (0.9*self.parametersMap["BeamEnergy_" + mode], 1.1*self.parametersMap["BeamEnergy_" + mode] )
        self.paramBounds["BeamSpread_" + mode] = (0.5*self.parametersMap["BeamSpread_" + mode], 1.5*self.parametersMap["BeamSpread_" + mode])


        self.LRpars = []
        self.LPpars = []

        self.paramDegree = self.settings.LayerSettings.getSettings("Parameterization_Degree") #[2,1,4]
        self.Nsublayers = self.settings.LayerSettings.getSettings( "N_sublayers") # [100, 20, 30]
        self.parameterizationNames = self.settings.LayerSettings.getSettings( "Parameterization") # [100, 20, 30]
        self.disc_scalings =  self.settings.LayerSettings.getSettings("Discretization_scaling")


        self.elementParamListList = []  # [[poly1, poly2, ...], ]
        self.paramNameToParameterizationMap = {}  # is a dict {parameter name : ElementPoly object}

        if not (len(self.Nsublayers) == len(self.paramDegree) == self.Nlayers):
            print("Nsublayers, polydegree, Nlayers list from setting : ", self.Nsublayers, self.paramDegree, self.Nlayers)
            raise Exception("The created SIMNRA object does not have matching layers as the program setting indicated.")

        for i in range(1, self.Nlayers + 1):
            parameterization_name = self.parameterizationNames[i-1]
            disc_scaling = self.disc_scalings[i-1]
            if disc_scaling=="Log" : 
                self.parametersMap["DiscLogBase_"+str(i)]= 4.0 
                self.paramBounds["DiscLogBase_"+str(i)] = (1.01, 10.0)
            self.parametersMap["Thickness_" + str(i)] = self.simNRA.Target.LayerThickness(i)
            self.paramBounds["Thickness_" + str(i)] = (0.1*self.simNRA.Target.LayerThickness(i), 2.0* self.simNRA.Target.LayerThickness(i))
            if self.lr[i - 1]:
                self.parametersMap["LayerRoughness_" + str(i)] = 0.01  # initialized with 1% of the thickness
                self.paramBounds["LayerRoughness_" + str(i)] = (0.0, 0.1)
                self.LRpars.append("LayerRoughness_" + str(i))

            if self.lp[i - 1]:
                self.parametersMap["PorosityFraction_" + str(i)] = 0.0
                self.parametersMap["PorosityDiameter_" + str(i)] = 0.01  # initialized with 1% of the thickness
                self.paramBounds["PorosityFraction_" + str(i)] = (0.0, 0.5)
                self.paramBounds["PorosityDiameter_" + str(i)] = (0.0, 0.1)  
                self.LPpars.append("PorosityFraction_" + str(i))
                self.LPpars.append("PorosityDiameter_" + str(i))

            el_param_list = []
            for j in range(1, self.NelPerLay[i - 1] + 1):
                el = str(self.simNRA.Target.ElementName(i, j)).strip()

                if parameterization_name == "Constant" : 
                    params = Constant  (str(i) + "_" + el)
                    params.setParameter("Cons_" + str(i) + "_" + el, self.simNRA.Target.ElementConcentration(i, j))
                elif parameterization_name == "Polynomial" : 
                    params = Polynomial(str(i) + "_" + el, self.paramDegree[i - 1])
                    params.setParameter("Poly_" + str(i) + "_" + el + "_" + str(0),  self.simNRA.Target.ElementConcentration(i, j))
                elif parameterization_name == "Erf-polynomial" : 
                    params = Erf_polynomial(str(i) + "_" + el, self.paramDegree[i - 1])
                    params.setParameter("Erf_" + str(i) + "_" + el + "_A", self.simNRA.Target.ElementConcentration(i, j))
                else : raise Exception("Uknown layer parameterization : ", parameterization_name)

                params.width = self.parametersMap["Thickness_" + str(i)]
                el_param_list.append(params)
                for parname in params.allParams: self.paramNameToParameterizationMap [parname] = params
                self.parametersMap.update(params.allParams)
                self.paramBounds.update(params.paramBounds)

            self.elementParamListList.append(el_param_list)

    def __del__(self):
        """Destructor"""
        del self.simNRA 
        try:
            os.remove(self.tempfile)
        except:
            print("Can not delete the temporary file : {}, but you can delete the file manually".format(self.tempfile))


    def createSIMNRAobject(self):
        """ 
        Create SIMNRA object from the settingsApp object
        """
        # Create a temporary directory within the system's temporary directory
        temp_folder = Path(tempfile.gettempdir()) / "AutoNRA_Temp"
        temp_folder.mkdir(parents=True, exist_ok=True)

        num = np.random.random()
        tempfile_path = str(temp_folder / (self.mode + "_" + str(num) + ".xnra"))
        reffile = self.settings.IBASettings.getIBASettings(self.mode, "Reference_file")
        if reffile =="" : 
            raise Exception("Can not create SIMNRA object. Please provide a valid reference file.")
        # Copy the reference file
        shutil.copy(reffile, tempfile_path)
        
        # Get absolute path of the temporary file
        self.tempfile = os.path.abspath(tempfile_path)
        # Initialize SIMNRA object
        self.simNRA = SIMNRA()
        self.simNRA.App.SimulationMode = 0
        
        # Set SimulationMode based on mode
        if "PIGE" in self.mode:
            self.simNRA.App.SimulationMode = 1
        
        # Open the temporary file with SIMNRA
        if not self.simNRA.App.Open(self.tempfile, -1):
            err = str(self.simNRA.App.LastMessage)
            raise SIMNRAError(err)
        
        # Configure SIMNRA calculations
        self.simNRA.Calc.ElementSpectra = True
        self.simNRA.Fit.Chi2Evaluation = 0
        
        # Configure layers in SIMNRA
        NlayerRef = self.simNRA.Target.NumberOfLayers #store the initial number of layer in the ref file
        NumLayers = self.settings.LayerSettings.getSettings("NumberOfLayers")
        ThicknessPerLayer = self.settings.LayerSettings.getSettings("ThicknessPerLayer")
        elemenAndConsPerLayers = self.settings.LayerSettings.getSettings("ElementsPerLayer")
        ElPerLayer = []
        ElConsPerLayer = []
        for el in elemenAndConsPerLayers :
            elList = []
            consList=[]
            for e in el : 
                elList.append(e[0])
                consList.append(e[1])
            if np.sum(consList)!=1.0 : consList = list(np.array(consList)/np.sum(consList)) 
            ElPerLayer.append(elList)
            ElConsPerLayer.append(consList)

        
        
        for i in range(NlayerRef + 1, NumLayers + NlayerRef + 1):
            self.simNRA.Target.AddLayer()
            self.simNRA.Target.AddElements(i, len(ElPerLayer[i - NlayerRef - 1]))
            self.simNRA.Target.SetLayerThickness(i, ThicknessPerLayer[i - NlayerRef - 1])
            
            for j, el in enumerate(ElPerLayer[i - NlayerRef - 1]):
                self.simNRA.Target.SetElementName(i, j + 1, el)
                self.simNRA.Target.SetElementConcentration(i, j + 1, ElConsPerLayer[i - NlayerRef - 1][j])
        
        # Delete layers from reference file
        for i in range(NlayerRef):
            self.simNRA.Target.DeleteLayer(1)#delete initial layers
        
        # Configure calculation settings
        calcSett = self.settings.SIMNRASettings.getAllSettings()
        self.simNRA.Calc.DualScattering = calcSett["DualScatterings"]
        self.simNRA.Calc.DualScatteringRoughness = 1 if calcSett["DualScatteringandRoughnessCalc"] == "Fast" else 0
        self.simNRA.Calc.Isotopes = calcSett["Isotopes"]
        self.simNRA.Calc.MultipleScattering = calcSett["MultipleScatterings"]
        self.simNRA.Calc.NuclearStoppingModel = 0 if calcSett["NuclearStoppingPowerData"] == "None" else 1
        
        # Set ScreeningModel based on settings
        self.simNRA.Calc.ScreeningModel = {
            "None": 0,
            "Andersen": 1,
            "L'Ecuyer": 2,
            "Universal": 3
        }[calcSett["RutherfordCSScreening"]]
        
        # Set StoppingModel based on settings
        self.simNRA.Calc.StoppingModel = {
            "Andersen/Ziegler": 0,
            "Ziegler/Biersack": 1,
            "ZB+KKK": 2,
            "User defined": 3,
            "SRIM": 4,
            "DPASS": 5
        }[calcSett["Estoppingpower"]]
        
        # Configure Straggling settings
        self.simNRA.Calc.Straggling = calcSett["Straggling"]
        self.simNRA.Calc.StragglingModel = {
            "Bohr": 1,
            "Chu": 2,
            "Chu+Yang": 3
        }[calcSett["ElossStragglingModel"]]
        self.simNRA.Calc.StragglingShape = 1 if calcSett["StragglingShape"] == "AsymmetricGaussian" else 0
        
        # Configure MultipleScatteringModel
        self.simNRA.Calc.MultipleScatteringModel = 0 if calcSett["MultipleScatteringModel"] == "Szilagyi" else 1
        
        # Configure CrossSecStraggling
        self.simNRA.Calc.CrossSecStraggling = 2 if calcSett["WeightingCSByStraggling"] == "Accurate" else 1
        
        # Configure SubstrateRoughnessDimension
        self.simNRA.Calc.SubstrateRoughnessDimension = 1 if calcSett["SubstrateRoughnessDimension"] == "2.5D" else 0
        
        # Set number of variations
        self.simNRA.Calc.NumberOfDVariations = calcSett["RoughnessNthicknessStep"]
        self.simNRA.Calc.NumberOfAngleVariations = calcSett["RoughnessNangularStep"]

    def checkThicknessRange(self):
        """ 
        Check if the thickness of the sample is too large such that the energy of the ion beam is less then the threshold self.EIonMin
        """        
        if self.checkRange:
            Ein = self.ionEnergy
            for i in range(1, self.simNRA.Target.NumberOfLayers + 1):
                dE = self.simNRA.Stopping.EnergylossInLayer(self.ionZ, self.ionMass, Ein, 1, i)
                Ein = Ein - dE
                if Ein < self.EIonMin:
                    raise ThicknessOutOfRangeError()

    def getLayerInfoFromSimnra(self):
        """ 
        Get Layer infor from SIMNRA, for debugging purpose.
        """        
        print("=================================")
        print("PSR :", self.simNRA.Setup.ParticlesSr)
        print("Call Offset : ", self.simNRA.Setup.CalibrationOffset)
        print("Call Linear : ", self.simNRA.Setup.CalibrationLinear)
        print("Call Quadratic : ", self.simNRA.Setup.CalibrationQuadratic)
        
        for i in range(1, self.simNRA.Target.NumberOfLayers + 1):
            print("Layer ", i, " Info : ")
            print("   Thickness : ", self.simNRA.Target.LayerThickness(i))
            
            eldict = {}
            for j in range(1, self.simNRA.Target.NumberOfElements(i) + 1):
                elname = str(self.simNRA.Target.ElementName(i, j)).strip()
                eldict[elname] = self.simNRA.Target.ElementConcentration(i, j)
            
            print("   Elemental concentration : ")
            print("         ", eldict)
            print("   Sums : " , np.sum(list(eldict.values())))
        
        print("")

    def getAllFittableParameters(self):
        """ 
        Get all fittable parameters, given all the setting self.settings
        """

        ret = {}        
        for p, val in self.parametersMap.items():
            if "PIGE" in self.mode:
                if ("ParticlesSr" in p) or ("Calib" in p):
                    continue
            ret[p] = val
        
        return ret
    
    def getAllFittableParameterBounds(self):
        """ 
        Get all fittable parameters, given all the setting self.settings
        """
        ret = {}        
        for p, val in self.paramBounds.items():
            if "PIGE" in self.mode:
                if ("ParticlesSr" in p) or ("Calib" in p):
                    continue
            ret[p] = val
        
        return ret

    def getSpectrumArray(self):
        self.simNRA.App.CalculateSpectrum()
        ret = list(self.simNRA.Spectrum.GetDataArray(2))
        return ret
    
    def getDepthProfile(self):
        xl = [0.0]
        elCons = {}
        # Flatten ElPerLayer and get unique elements
        allEl = list(self.elZdict.keys())
        # Initialize element concentrations dictionary
        for el in allEl: elCons[el] = [0.0] * self.simNRA.Target.NumberOfLayers
        # Populate xl and elCons with layer thicknesses and element concentrations
        for i in range(1, self.simNRA.Target.NumberOfLayers + 1):
            xl.append(self.simNRA.Target.LayerThickness(i))
            for j in range(1, self.simNRA.Target.NumberOfElements(i) + 1):
                el = str(self.simNRA.Target.ElementName(i, j)).strip()
                elCons[el][i - 1] = self.simNRA.Target.ElementConcentration(i, j)
        return [xl, elCons]

    def setLayerProperties(self, myDict):
        for key, val in myDict.items():
            if key not in self.parametersMap: raise ParameterNotFoundError(key)
            self.parametersMap[key] = val

        for key, val in self.parametersMap.items():
            if key.startswith("ParticlesSr_"):
                if val < 0.0: raise ParameterNegativeError(key)
                self.simNRA.Setup.ParticlesSr = val
            elif "Thickness" in key:
                if val < 0.0: raise ParameterNegativeError(key)
            elif key.startswith("Calib_"):
                if "Offset" in key:
                    self.simNRA.Setup.CalibrationOffset = val
                elif "Linear" in key:
                    self.simNRA.Setup.CalibrationLinear = val
                elif "Quadratic" in key:
                    self.simNRA.Setup.CalibrationQuadratic = val
                else:
                    raise Exception("Unknown calibration parameters!")
            elif key.startswith("FWHM"): self.simNRA.Setup.DetectorResolution = val
            elif  key.startswith("BeamEnergy"): self.simNRA.Setup.Energy = val
            elif key.startswith("BeamSpread"): self.simNRA.Setup.Beamspread = val
            elif "DiscLogBase_" in key:
                if val<1.0 : raise Exception("Log base of the discretization can not be less than 1")
            elif "Roughness_" in key:
                pass
            elif "Porosity" in key:
                pass
            elif ("Poly_" in key) or ("Erf_" in key) or ("Cons_" in key):
                self.paramNameToParameterizationMap[key].setParameter(key, val)
            else:
                print(self.mode, key, val)
                raise Exception("Problem setting layer parameters. Parameter name : ", key, "is not recognized.")

        self.__calcElementalConcentration()

        k = 0
        if self.first:  # create all the sublayers first
            for i in range(self.Nlayers):
                for j in range(self.Nsublayers[i]):
                    self.simNRA.Target.AddLayer()
                    Nel = self.NelPerLay[i]
                    lay = self.fullLayerEC[i][j]  # [thick, xpos, [0.1, ...]]
                    self.simNRA.Target.AddElements(k + j + 1 + self.Nlayers, Nel)
                    self.simNRA.Target.SetLayerThickness(k + j + 1 + self.Nlayers, lay[0])
                    for l in range(Nel):
                        self.simNRA.Target.SetElementName(k + j + 1 + self.Nlayers, l + 1, self.ElPerLayer[i][l])
                        self.simNRA.Target.SetElementConcentration(k + j + 1 + self.Nlayers, l + 1, lay[2][l])
                k += self.Nsublayers[i]

            for i in range(self.Nlayers):
                self.simNRA.Target.DeleteLayer(1)
            
            self.first = False
            totNumLay = np.sum(self.Nsublayers)
            
            if totNumLay != self.simNRA.Target.NumberOfLayers:
                raise Exception("Total number of layers does not match with SIMNRA layers!")
            
            self.phyLayerToSubLayerMap = {}  # this store the map that connect the index of physical layers to the index of the corresponding first sublayer
            ind = 1
            for i in range(1, self.Nlayers + 1):
                self.phyLayerToSubLayerMap[i] = ind
                fl = self.lr[i - 1]
                self.simNRA.Target.SetHasLayerRoughness(i, fl)
                ind += self.Nsublayers[i - 1]

            self.phyLayerToSubLayerMap2 = {}
            shift = 0
            for i in range(1, self.Nlayers + 1):
                mylist = []
                fl = self.lp[i - 1]
                for j in range(1, self.Nsublayers[i - 1] + 1):
                    mylist.append(j + shift)
                    self.simNRA.Target.SetHasLayerPorosity(j + shift, fl)
                self.phyLayerToSubLayerMap2[i] = mylist
                shift += self.Nsublayers[i - 1]

        else:
            for i in range(self.Nlayers):
                for j in range(self.Nsublayers[i]):
                    Nel = self.NelPerLay[i]
                    lay = self.fullLayerEC[i][j]  # [thick, xpos, [0.1, ...]]
                    self.simNRA.Target.SetLayerThickness(k + j + 1, lay[0])
                    for l in range(Nel):
                        self.simNRA.Target.SetElementConcentration(k + j + 1, l + 1, lay[2][l])
                k += self.Nsublayers[i]

        for par in self.LRpars:
            lay = par.split("_")[-1]
            self.simNRA.Target.SetLayerRoughness(self.phyLayerToSubLayerMap[int(lay)],
                                                 self.parametersMap[par] * self.parametersMap["Thickness_" + str(lay)])

        for par in self.LPpars:
            lay = int(par.split("_")[-1])
            mylist = self.phyLayerToSubLayerMap2[lay]
            for l in range(len(mylist)):
                j = mylist[l]
                if "Fraction" in par:
                    self.simNRA.Target.SetPorosityFraction(j, self.parametersMap[par])
                else:
                    self.simNRA.Target.SetPoreDiameter(j, self.parametersMap[par] * self.parametersMap["Thickness_" + str(lay)])
        self.checkThicknessRange()

    def __discretisize(self):
        shift = 0.0
        widthList = [] #list of list of width of sublayers
        xlist = [] #list of list of the center of svlayers
        shift = 0.0
        for i in range(self.Nlayers): 
            thick = self.parametersMap["Thickness_" + str(i + 1)]
            Ndesc = self.Nsublayers[i]

            discScaling = self.disc_scalings[i]
            if discScaling == "Linear":  xpos = np.linspace(0.0, 1.0, Ndesc + 1) * thick
            elif discScaling == "Log": 
                xpos = np.logspace(-3, 0, Ndesc + 1, base=self.parametersMap["DiscLogBase_"+str(i+1)]) * thick
                xrem = xpos[0]
                for k in range(len(xpos)): xpos[k] = xpos[k] - xrem
                xpos[-1] = xpos[-1] + xrem
            else : raise Exception("Uknown disctretization scaling.")

            wl = [] #the list of width of each sublayer
            xl = [] # list of the center of each sublayer
            for j in range(Ndesc): 
                wl.append(xpos[j + 1] - xpos[j])
                xl.append(shift + 0.5 * (xpos[j + 1] + xpos[j]))
            assert len(xl) == len(wl) == Ndesc
            widthList.append(wl) 
            xlist.append(xl)
            shift = shift + thick

        self.fullLayerEC = [] # the list of layer elemental concentration per sublayers : [ [ [width_1, pos_1, [conc_1, conc_2, ..]], ..], ... ]
        for i in range(self.Nlayers): 
            layList = [] 
            for j in range(self.Nsublayers[i]): layList.append([widthList[i][j], xlist[i][j], []])
            self.fullLayerEC.append(layList)

    def __calcElementalConcentration(self):
        self.__discretisize()
        for i in range(self.Nlayers):
            elementParamList = self.elementParamListList[i] #[param_el1, param_el2, ..]
            for j in range(self.Nsublayers[i]):
                lay = self.fullLayerEC[i][j] # [width, position, []]
                consList = []
                for k in range(len(elementParamList)):
                    cons = elementParamList[k].func(lay[1])
                    if cons < 0.0:
                        raise ParameterNegativeError(self.ElPerLayer[i][k])
                        # print("  Warning : element ", self.ElPerLayer[i][k], "has negative concentration. Will set it to zero.")
                        # cons = 0.0
                    consList.append(cons)

                conSum = np.sum(consList)
                if self.checkConSum:
                    if np.abs(conSum - 1.0) > 0.0001:
                        raise LayerNotNormalizedError(j)
                self.fullLayerEC[i][j][2] = np.array(consList) / conSum
