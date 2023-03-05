"""
This module is responsible for the namespaces of the measurement quantities.

It includes several classes to handle different namespaces. See details in their respective documentations.

Contact person: Stefan Riedmaier
Creation date: 22.06.2020
Python version: 3.8
"""


# -- CLASSES -----------------------------------------------------------------------------------------------------------

class QuantityNamespaceHandler:
    """
    This class is a parent class for namespace handlers.

    It includes basic functions for the dictionary-based mappings that can be overwritten.
    """

    def __init__(self):
        """
        This method initializes a new class instance.
        """

        self.name_mapper_dict = dict()

    def quantity_name_mapper(self, reference_quantity_name):
        """
        This function returns the quantity name in a target namespace corresonding to a reference quantity name.

        If no matching name is found, an index error will be raised.

        :param str reference_quantity_name: name in the reference namespace
        :return: name in target namespace
        :rtype: str
        """

        target_quantity_name = self.name_mapper_dict.get(reference_quantity_name)

        if target_quantity_name is None:
            raise IndexError("quantity not defined in mapping")

        return target_quantity_name

    def quantity_name_mapper_none(self, reference_quantity_name):
        """
        This function returns the quantity name in a target namespace corresonding to a reference quantity name.

        If no matching name is found, None will be returned.

        :param str reference_quantity_name: name in the reference namespace
        :return: name in target namespace
        :rtype: str
        """

        target_quantity_name = self.name_mapper_dict.get(reference_quantity_name)

        return target_quantity_name


class CarMakerNameSpaceHandler(QuantityNamespaceHandler):
    """
    This child class is responsible for the CarMaker quantity name space used as reference.
    """

    def quantity_name_mapper(self, reference_quantity_name):
        """
        This function overwrittes the dictionary mappings, since the CarMaker namespace is selected as reference.

        :param reference_quantity_name: name in the reference namespace
        :return: name in the reference namespace
        :rtype: str
        """

        return reference_quantity_name


class R79VehicleMdfNamespaceHandler(QuantityNamespaceHandler):
    """
    This child class is responsible for the quantity name space of the physical UNECE-R79 vehicle tests with MDF files.
    """

    def __init__(self):
        """
        This method initializes a new class instance.
        """

        super().__init__()

        # quantity name mapping from reference namespace (keys) to target namespace (values in measurement files)
        self.name_mapper_dict = {
            'Car.ax': 'INS_Accel_X_Raw',  # unfiltered raw longitudinal acceleration
            # 'Car.ay': 'INS_Accel_Y_Raw',  # unfiltered raw lateral acceleration
            'Car.ay': 'INS_Accel_Y_Filtered',  # actual measured lateral acceleration
            'Car.ay_ref': 'INS_Accel_Y_Reference',  # based on map radius and actual measured velocity
            'Car.az': 'INS_Accel_Z_Raw',
            'Car.vx': 'INS_Vel_X_Raw',
            'Car.vy': 'INS_Vel_Y_Raw',
            'Car.v': 'INS_Vel_X_Raw',
            'LatCtrl.LKAS.IsActive': 'BUS_ACSF_Status',
            'LatCtrl.LKAS.SwitchedOn': 'BUS_ACA_QF_Aktiviert',
            'LatCtrl.DevDist': 'D2CL',
            'LatCtrl.DistToRight': 'D2RL',  # distance to left line from the center of the vehicle
            'LatCtrl.DistToLeft': 'D2LL',  # distance to right line from the center of the vehicle
            'LatCtrl.LKAS.CurveXY_trg': 'Curvature',
            'LatCtrl.LKAS.CurveXY_trg_BUS': 'BUS_BV_Curvature_C',
            'D2LL': 'D2LL',  # distance to left line from the left edge of the vehicle
            'D2RL': 'D2RL',  # distance to right line from the right edge of the vehicle
            'D2LL_SHAPE_BUS': 'BUS_D2LL_SHAPE',
            'D2RL_SHAPE_BUS': 'BUS_D2RL_SHAPE'
        }


class R79VehicleMatNamespaceHandler(QuantityNamespaceHandler):
    """
    This child class is responsible for the quantity name space of the physical UNECE-R79 vehicle tests with MAT files.
    """

    def __init__(self):
        """
        This method initializes a new class instance.
        """

        super().__init__()

        # quantity name mapping from reference namespace (keys) to target namespace (values)
        self.name_mapper_dict = {
            'Car.ax': 'Acc_Hor_X_POI5',
            'Car.ay': 'Acc_Hor_Y_POI5',
            'Car.az': 'Acc_Hor_Z_POI5',
            'Car.vx': 'INS_Vel_Hor_X_POI5',
            'Car.vy': 'INS_Vel_Hor_Y_POI5'
        }


class R79ParameterNamespaceHandler(QuantityNamespaceHandler):
    """
    This child class is responsible for the parameter name space of the UNECE-R79 examples.
    """

    def __init__(self):
        """
        This method initializes a new class instance.
        """

        super().__init__()

        # mapping from parameter namespace (keys) to reference quantity namespace (values)
        self.name_mapper_dict = {
            # '$Ego_Init_Velocity': 'Car.v',
            # '$Radius': 'Curvature'
        }
