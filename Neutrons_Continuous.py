import sys
import ROOT
from ROOT import TMath, TCanvas, TRandom3, TH1F, TH2D, TGraph, TF1
from glob import glob
import math
from ROOT import TLorentzVector, TVector3, TGraphErrors
from array import array
from math import sqrt
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

# Don't show plots
ROOT.gROOT.SetBatch(1)

# to do

# - are energy errors too smal for 5MeV case? How come they are not vissible?
# - reject bad chi^2 fits?
# - plain formatting / histogram styles
# - add angular miss-measurement if time
# - add extraction of trends, i.e. resolution(with error) versus n-photon

# FIXED: check proton energy spectrum --> too high energy tail!
# FIXED: most important: why is there a problem with more than 1 iterations?????? --> distortion of energy spectrum! --> neutron beam was boosted in subroutine!

neutron_mass = 939.56556081  # [MeV]
proton_mass = 938.27201323  # MeV
pi = TMath.Pi()
xaxis = TVector3(1, 0, 0)
todeg = 180.0 / pi

sigma_phi_radians = 2.0 / todeg
sigma_theta_radians = 2.0 / todeg
sigma_phi_deg = 2.0
sigma_theta_deg = 2.0


def Style():
    ROOT.gROOT.SetStyle("Plain")  # Switches off the ROOT default style
    # this makes everything black and white, removing the default red border on images
    ROOT.gROOT.UseCurrentStyle()
    # forces the style chosen above to be used, not the style the rootfile was made with
    ROOT.gROOT.ForceStyle()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1111)
    ROOT.gStyle.SetStatY(0.9)
    ROOT.gStyle.SetStatX(0.9)
    ROOT.gStyle.SetStatW(0.20)
    ROOT.gStyle.SetStatH(0.10)


def main():

    # Style()
    # number of measurements used to estimate resolution
    iterationString = input("Enter number of iterations: ")
    iterations = int(iterationString)

    counter = 0
    c1 = TCanvas("Results")
    random_engine = TRandom3()

    # neutron beam kinetic energy [MeV]
    neutron_energies = [5.5]
    neutronString = input("Enter number of neutrons detected: ")
    n_neutrons = int(neutronString)

    smearing = input(
        "Do you want detector smearing? (y/n) ").lower().strip() == 'y'

    print("Smearing set to: ", smearing)

    fileNameString = input("Enter file name for saving: ")

    gas = ["Propane"]

    h_angle_residual = TH1F(
        "neutron_angle_residual", "neutron angle residual (degrees)", 100, -
        10, 10
    )
    h_energy_residual = TH1F(
        "neutron_energy_residual",
        "neutron energy residual (fraction of neutron energy)",
        100,
        -0.4,
        0.4,
    )

    h_phi = TH1F("proton_phi", "proton phi", 100, -3.2, 3.2)
    h_theta = TH1F("proton_theta", "proton theta", 100, 0.0, 3.2)
    h_angle_xaxis = TH1F(
        "proton_angle_x", "proton angle wrt x-axis", 100, -3.2, 3.2)
    h_cos_angle_xaxis = TH1F(
        "proton_cos_angle_x", "proton cos(angle wrt x-axis)", 100, -3.2, 3.2
    )
    h_energy = TH1F("proton_energy", "proton energy (MeV)", 100, 0, 6)
    h_energy_xaxis = TH1F(
        "proton_energy_x",
        "proton angle wrt x-axis, wheighted by energy",
        100,
        -3.2,
        3.2,
    )
    h_theta_vs_phi = TH2D(
        "proton_theta_vs_phi",
        ";#Theta (degrees);#Phi (degrees);recoils per bin",
        25,
        0,
        180.0,
        25,
        -180.0,
        180.0,
    )
    h_theta_vs_phi_weighted = TH2D(
        "proton_theta_vs_phi_weighted",
        ";#Theta (degrees);#Phi (degrees);energy per bin (MeV)",
        25,
        0,
        180.0,
        25,
        -180.0,
        180.0,
    )

    for kinetic_energy in neutron_energies:

        # Lists for CSVs for training
        recoilList = []
        h_angle_residual.Reset()
        h_energy_residual.Reset()

        NAME = "{0:.2f}_MeV".format(kinetic_energy)

        for iteration in tqdm(range(iterations)):
            kinetic_energy_random = random.uniform(0.0, kinetic_energy)
            E_n = neutron_mass + kinetic_energy_random
            p_n = sqrt(E_n**2 - neutron_mass**2)

            # incoming neutron beam is in x-direction
            p3_n = TVector3(p_n, 0, 0)
            incoming_neutron = TLorentzVector(p3_n, E_n)

            debug = False
            counter = counter+1
            if iteration == 10:
                debug = True
            #
            # the actual simulation happens here
            #

            # estimate how well we can measure incoming neutron beam direction based on n neutrons
            true_protons = proton_scattering(
                random_engine, n_neutrons, incoming_neutron, gas
            )

            reco_protons = proton_detection(
                random_engine, true_protons, smearing)

            recoilSet = []

            for proton in reco_protons:
                recoilSet.append((proton.E() - proton.M()) / 5.2)
                recoilSet.append(proton.Theta() / 3.2)
                recoilSet.append((proton.Phi() + 3) / 6.3)

            recoilSet.append(kinetic_energy_random / 5)
            recoilList.append(recoilSet)

            energy, angle = find_beam_direction(
                reco_protons, kinetic_energy_random, debug)
            h_angle_residual.Fill((angle - 0.0))
            h_energy_residual.Fill(
                (energy - kinetic_energy_random) / kinetic_energy_random)

            if debug is True:
                # make "event displays" for one event (consisting of n proton recoils)
                for proton in reco_protons:
                    energy = proton.E() - proton.M()  # kinetic energy
                    h_theta.Fill(proton.Theta())
                    h_phi.Fill(proton.Phi())
                    h_angle_xaxis.Fill(proton.Angle(xaxis))
                    h_cos_angle_xaxis.Fill(TMath.Cos(proton.Angle(xaxis)))
                    h_energy_xaxis.Fill(proton.Angle(xaxis), energy)
                    h_energy.Fill(energy)
                    h_theta_vs_phi.Fill(
                        proton.Theta() * todeg, proton.Phi() * todeg
                    )
                    h_theta_vs_phi_weighted.Fill(
                        proton.Theta() * todeg, proton.Phi() * todeg, energy
                    )

                for histogram in [
                    h_phi,
                    h_theta,
                    h_angle_xaxis,
                    h_energy_xaxis,
                    h_energy,
                    h_cos_angle_xaxis,
                ]:
                    histogram.Draw()
                    c1.Print("Plots/Generator/" +
                             histogram.GetName() + ".png")

                for histogram in [h_theta_vs_phi, h_theta_vs_phi_weighted]:
                    histogram.GetXaxis().SetTitleOffset(1.6)
                    histogram.GetYaxis().SetTitleOffset(1.6)
                    histogram.Draw("lego1")
                    c1.Print("Plots/Generator/" +
                             histogram.GetName() + "_lego.png")

                for histogram in [h_theta_vs_phi, h_theta_vs_phi_weighted]:
                    histogram.Draw("surf3polz")
                    c1.Print("Plots/Generator/" +
                             histogram.GetName() + "_surf3polz.png")
        c1.cd()
        for histogram in [h_angle_residual, h_energy_residual]:
            histogram.Draw()
            histogram.Fit("gaus", "Q")
            c1.Print("Plots/Generator/" +
                     histogram.GetName()
                     + "_"
                     + str(kinetic_energy)
                     + "MeV_"
                     + str(n_neutrons)
                     + "neutrons.png"
                     )
    # fit = gr_dr.GetFunction("myfit");
    # Double_t chi2 = fit->GetChisquare();
    # energy = fit.GetParameter(0);
    # angle  = fit.GetParameter(1);

        path = 'Data/Continuous/' + neutronString + '_Recoils/' + fileNameString
        recoilArray = pd.DataFrame(recoilList)
        recoilArray.to_pickle(path + '_{}.pkl'.format(NAME))

    print("Done!")
    print("Recoil list shape:", recoilArray.shape)


def find_beam_direction(protons, neutron_energy, debug=False):
    plot = False

    phi = []
    theta = []
    energy = []
    energy_sigma = []
    dr = []
    dr_sigma = []

    for proton in protons:
        kin_energy = proton.E() - proton.M()
        phi += [proton.Phi()]
        theta += [proton.Theta()]
        energy += [kin_energy]
        energy_sigma += [sigma_energy(kin_energy)]
        dr += [
            todeg * proton.Angle(TVector3(1, 0, 0))
        ]  # angle w.r.t. beam axis --> FIXME: make beam axis a variable here and in other parts of code!
        dr_sigma += [
            sqrt(sigma_phi_deg**2 + sigma_theta_deg**2)
        ]  # FIXME: just use a two degree sigma for now, study dependence later to see how much it affects final resolution
        if debug is True:
            print(
                "proton kin, phi, theta, dr, sigma_e =",
                kin_energy,
                proton.Phi(),
                proton.Theta(),
                proton.Angle(TVector3(1, 0, 0)),
                sigma_energy(kin_energy),
            )

    a_phi = array("f", phi)
    a_theta = array("f", theta)
    a_energy = array("f", energy)
    a_energy_sigma = array("f", energy_sigma)
    a_dr = array("f", dr)
    a_dr_sigma = array("f", dr_sigma)

    gr_phi = TGraph(len(a_phi), a_phi, a_energy)
    gr_theta = TGraph(len(a_theta), a_theta, a_energy)
    gr_dr = TGraphErrors(len(a_dr), a_dr, a_energy, a_dr_sigma, a_energy_sigma)

    gr_dr.GetXaxis().SetTitle("recoil angle with respect to neutron beam (degrees)")
    gr_dr.GetYaxis().SetTitle("recoil energy (MeV)")

    gr_phi.SetMarkerColor(4)
    gr_phi.SetMarkerStyle(21)
    gr_theta.SetMarkerColor(4)
    gr_theta.SetMarkerStyle(21)
    gr_dr.SetMarkerColor(4)
    gr_dr.SetMarkerStyle(21)

    if debug is True:
        c2 = TCanvas("c2", "A Simple Graph Example", 200, 10, 700, 500)
        c2.cd()
        gr_phi.Draw("AP")
        c2.Print("Plots/Generator/" + "graph_phi.png")
        gr_theta.Draw("AP")
        c2.Print("Plots/Generator/" + "graph_theta.png")
        c2.DrawFrame(0, 0, todeg * pi / 2, 1.1 * neutron_energy)
        gr_dr.Draw("AP")

    #  EXT PARAMETER                                   STEP         FIRST
    #  NO.   NAME      VALUE            ERROR          SIZE      DERIVATIVE
    #   1  c1           7.07283e-02   8.57732e-02   4.11493e-06  -7.01509e-04
    #   2  c2          -1.35491e+00   1.80217e-01   3.83497e-06  -7.63102e-04
    #   3  c3           5.77298e-01   9.09558e-02   3.22840e-06  -8.78497e-04

    myfit = TF1(
        "myfit",
        "[0]*(1.0 + 0.0707283*0.017453*(x-[1]) - 1.35491*(0.017453*(x-[1]))**2 + 0.577298*(0.017453*(x-[1]))**3)",
        0,
        pi / 2,
    )

    # Now, we can set parameter names (optional) and parameter start values (mandatory).

    myfit.SetParName(0, "energy")
    myfit.SetParName(1, "angle")
    myfit.SetParameter(0, 1)
    myfit.SetParameter(1, 0)

    gr_dr.Fit("myfit", "Q")

    if debug is True:
        c2.Print("Plots/Generator/" + "graph_dr.png")

    fit = gr_dr.GetFunction("myfit")
    # Double_t chi2 = fit->GetChisquare();
    energy = fit.GetParameter(0)
    angle = fit.GetParameter(1)
    # print("fitted energy=", energy, "fitted angle =", angle)
    # Root > Double_t e1 = fit->GetParError(1);

    return energy, angle


def proton_scattering(randeng, n_neutrons, incoming_neutron, gas):

    # range ["Propane"]= { 1.0:1.921E-03,
    #                     2.0:6.160E-03,
    #                     3.0:1.242E-02,
    #                     4.0:1.242E-02,
    #                     5.0:2.056E-02 }
    protons = []

    # proton at rest in lab frame
    p4p_lab = TLorentzVector(0, 0, 0, proton_mass)
    p4n_lab = TLorentzVector(incoming_neutron)

    # boots everything to COM
    com = TLorentzVector(p4p_lab + p4n_lab)
    boost = com.BoostVector()
    # print "boost = ", boost.x(), boost.y(), boost.z()
    inv_boost = TVector3(-boost.x(), -boost.y(), -boost.z())

    p4p_lab.Boost(inv_boost)
    p4n_lab.Boost(inv_boost)

    for n in range(n_neutrons):
        rnd = random_unit_vector(randeng)
        p4p_lab_final = TLorentzVector(p4p_lab)
        p4p_lab_final.SetTheta(rnd.Theta())
        p4p_lab_final.SetPhi(rnd.Phi())
        p4p_lab_final.Boost(boost)
        protons += [p4p_lab_final]
    return protons


def sigma_energy(kinetic_energy):

    # FIXME: Is this realistic, or do we need a constant term?
    #         2.65% at 1   MeV.
    #         8.84% at 100 keV
    # --> quite a bit better than 5% flat energy resolution

    IonizationEnergy = 24.0e-6  # (MeV)
    IonizationFraction = 0.3
    N_ionization = (kinetic_energy * IonizationFraction) / IonizationEnergy

    if N_ionization > 0:
        sigma_energy = kinetic_energy * 0.20 * sqrt(220 / N_ionization)
    else:
        sigma_energy = kinetic_energy
    return sigma_energy


def proton_detection(random_engine, true_protons, smear):
    detected = []

    for proton in true_protons:
        # random energy miss-measurment, 10% flat for now
        mass = proton.M()
        kin_energy = proton.E() - mass
        direction = TVector3(proton.Vect())

        if (smear == True):
            sigma = sigma_energy(kin_energy)
            measured_energy = random_engine.Gaus(kin_energy, sigma)
            measured_phi = direction.Phi() + random_engine.Gaus(0, sigma_phi_radians)
            measured_theta = direction.Theta() + random_engine.Gaus(0, sigma_theta_radians)
        else:
            measured_energy = kin_energy
            measured_phi = direction.Phi()
            measured_theta = direction.Theta()

        if measured_energy <= 0:
            measured_energy = 0

        new_energy = mass + measured_energy

        direction.SetPhi(measured_phi)
        direction.SetTheta(measured_theta)
        direction.SetMag(sqrt(new_energy**2 - mass**2)
                         )  # keeping theta and phi

        smeared_proton = TLorentzVector(direction, new_energy)

        detected += [smeared_proton]

    return detected


# taken from http://indra.in2p3.fr/KaliVedaDoc/src/KVPosition.cxx.html#OAsxuB
# FIXME: double check or validate that it's really isotropic


def random_unit_vector(rnd):
    dummy = TVector3(1.0, 0, 0)  # unit vector
    fPhi = rnd.Uniform(2 * pi) - pi
    fTheta = TMath.ACos(rnd.Uniform(-1.0, 1.0))
    dummy.SetTheta(fTheta)
    dummy.SetPhi(fPhi)
    return dummy


main()
