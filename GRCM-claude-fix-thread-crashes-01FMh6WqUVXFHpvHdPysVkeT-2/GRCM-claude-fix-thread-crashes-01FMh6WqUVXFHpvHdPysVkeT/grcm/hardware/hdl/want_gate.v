// ============================================================
// Want Modulation Gate
// Implements γψ where γ = 0.2 + 0.4*sigmoid(align)
// ============================================================

module want_gate #(
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,

    // Inputs
    input wire [DATA_WIDTH-1:0] psi,
    input wire [DATA_WIDTH-1:0] desire,
    input wire [15:0] align,  // Pre-computed alignment

    // Output
    output reg [DATA_WIDTH-1:0] gamma_psi,

    // Control
    input wire enable,
    output reg valid
);

    // Gamma calculation
    reg [15:0] gamma;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gamma <= 16'h0;
            gamma_psi <= 32'h0;
            valid <= 1'b0;
        end else if (enable) begin
            // γ = 0.2 + 0.4 * sigmoid(align)
            gamma <= compute_gamma(align);

            // γ * ψ
            gamma_psi <= multiply_gamma(psi, gamma);

            valid <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

    // Compute gamma (simplified sigmoid)
    function [15:0] compute_gamma;
        input [15:0] alignment;
        reg [15:0] sigmoid_val;
        begin
            // Simplified: sigmoid ≈ 0.5 + align/4 (linearized)
            sigmoid_val = 16'h2000 + (alignment >>> 2);  // 0.5 in fixed-point

            // γ = 0.2 + 0.4 * sigmoid
            compute_gamma = 16'h0CCC + ((sigmoid_val * 16'h1999) >>> 14);  // 0.2 + 0.4*sigmoid
        end
    endfunction

    // Multiply psi by gamma
    function [DATA_WIDTH-1:0] multiply_gamma;
        input [DATA_WIDTH-1:0] psi_val;
        input [15:0] gamma_val;
        reg signed [DATA_WIDTH-1:0] result;
        begin
            result = ($signed(psi_val) * $signed(gamma_val)) >>> 14;  // Fixed-point
            multiply_gamma = result;
        end
    endfunction

endmodule
