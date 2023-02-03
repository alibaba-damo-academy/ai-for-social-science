import os
import matplotlib.pyplot as plt
import numpy as np


class env_record(object):
    """
    Record the current environment state (Used for plotting)
    """

    def __init__(self, record_start_epoch=0, averaged_stamp=1, mode='dis_continous'):
        self.revenue_list = []
        self.efficient_list = []

        self.avg_efficient_history = []
        self.avg_revenue_history = []

        # The time to start record revenue after exploration
        self.record_start_epoch = record_start_epoch

        self.averaged_stamp = averaged_stamp  # The range that revenue is averaged over

        self.mode_list = ['dis_continous', 'incremental']
        self.mode = mode
        self.tmp_sum = 0

        self.efficiency_agt_name = None
        self.store_true_value = 0
        self.efficiency_max_true_value = 0

    def record_efficiency(self, allocation, true_value, epoch,agent_name, end_flag=False):
        if epoch >= self.record_start_epoch:
            self.efficiency_max_true_value = max(self.efficiency_max_true_value, true_value)

            if allocation == 1:
                # store the agent name for future use
                self.efficiency_agt_name = agent_name

                self.store_true_value = true_value

            if end_flag:
                # check the efficiency is reached or not
                if (self.efficiency_agt_name is None) or (self.efficiency_max_true_value > self.store_true_value):
                    # not the efficiency
                    self.efficient_list.append(0.0)
                else:
                    self.efficient_list.append(1.0)

                self.update_efficient_list()
                # init
                self.efficiency_max_true_value = 0
                self.store_true_value = 0
                self.efficiency_agt_name = None

    def update_efficient_list(self):

        if len(self.efficient_list) == self.averaged_stamp:
            self.avg_efficient_history.append(

                sum(self.efficient_list) * 1.0 / self.averaged_stamp
            )
            # mode 1:  incremental
            if self.mode == 'incremental':
                self.efficient_list = self.efficient_list[1:]
            # mode 2:  dis-continuous
            elif self.mode == 'dis_continous':
                self.efficient_list = []
            else:
                self.efficient_list = []
        else:
            # less than averaged stamp

            return

        return



    def adjust_mode(self, mode_id=0):
        self.mode = self.mode_list[mode_id]

    def record_revenue(self, allocation, pay, epoch, pay_sign=-1, mode='second_price', end_flag=False):
        if epoch >= self.record_start_epoch:
            if mode == 'pay_by_submit':
                self.tmp_sum += pay * pay_sign
                if end_flag:
                    self.revenue_list.append(self.tmp_sum)
                    self.update_revenue_list()
                    self.tmp_sum = 0
            elif mode == 'multi_item': 
                self.tmp_sum += np.sum(pay) * pay_sign
                if end_flag:
                    self.revenue_list.append(self.tmp_sum)
                    self.update_revenue_list()
                    self.tmp_sum = 0
            else:
                # pay sign ==-1 ---> pay is negative
                # current_revenue = pay * pay_sign
                # self.revenue_list.append(current_revenue)
                # self.update_revenue_list()
                
                print('current_revenue', current_revenue)
                if allocation == 1:
                    current_revenue = pay * pay_sign

                    self.revenue_list.append(current_revenue)
                    self.update_revenue_list()

    def get_last_avg_revenue(self):
        if len(self.avg_revenue_history) == 0:
            print(
                'not record on env avg revenvue (perhaps due to not record revenue during exploration, can be changed by decrease the record_start_epoch number)')
            return 0

        return self.avg_revenue_history[-1]

    def update_revenue_list(self):

        if len(self.revenue_list) == self.averaged_stamp:
            self.avg_revenue_history.append(

                sum(self.revenue_list) * 1.0 / self.averaged_stamp
            )
            # mode 1:  incremental
            if self.mode == 'incremental':
                self.revenue_list = self.revenue_list[1:]
            # mode 2:  dis-continuous
            elif self.mode == 'dis_continous':
                self.revenue_list = []
            else:
                self.revenue_list = []
        else:
            # less than averaged stamp

            return

        return

    def build_path(self, save_dir):
        # check path exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return

    def plot_avg_revenue(self, args, path='./', folder_name='', figure_name='', mechanism_name='second_price',
                         plot_y_range=None):

        if len(self.avg_revenue_history) > 0:
            output_path = os.path.join(os.path.join(path, args.folder_name), folder_name)
            self.build_path(output_path)

            plt.plot([x for x in range(len(self.avg_revenue_history))], self.avg_revenue_history, label=mechanism_name)
            plt.title('averaged platform revenue')
            if plot_y_range is not None:
                plt.ylim(plot_y_range[0], plot_y_range[1])

            plt.legend()
            print(os.path.join(output_path, figure_name + '_revenue_avg_on_' + str(self.averaged_stamp)))
            plt.savefig(os.path.join(output_path, figure_name + '_revenue_avg_on_' + str(self.averaged_stamp)))
            plt.show()
            plt.close()



    def plot_avg_efficiency(self, args, path='./', folder_name='', figure_name='', mechanism_name='second_price',
                         plot_y_range=None):

        if len(self.avg_efficient_history) > 0:
            output_path = os.path.join(os.path.join(path, args.folder_name), folder_name)
            self.build_path(output_path)

            plt.plot([x for x in range(len(self.avg_efficient_history))], self.avg_efficient_history, label=mechanism_name)
            plt.title('averaged platform efficiency')
            if plot_y_range is not None:
                plt.ylim(plot_y_range[0], plot_y_range[1])

            plt.legend()
            print(os.path.join(output_path, figure_name + '_efficiency_avg_on_' + str(self.averaged_stamp)))
            plt.savefig(os.path.join(output_path, figure_name + '_efficiency_avg_on_' + str(self.averaged_stamp)))
            plt.show()
            plt.close()
